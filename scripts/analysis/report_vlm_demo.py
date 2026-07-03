#!/usr/bin/env python3
"""
Gera um relatório HTML demonstrando o comportamento de cada VLM em duas tarefas:
  1. OCR — extração de texto de uma única imagem
  2. Similaridade — comparação de um par de imagens (igual e diferente)

Salva os resultados de inferência em JSON e gera o HTML a partir dele,
permitindo regenerar o HTML sem re-executar os modelos.

Uso:
  python scripts/analysis/report_vlm_demo.py \\
      --image-dir /mnt/data/la-cdip/data \\
      --image-a microbiological_associates_invoice/87103742.tif \\
      --image-b-same microbiological_associates_invoice/87104266.tif \\
      --image-b-diff rjreynolds_costumer_relations/518294700+-4701.tif \\
      --models internvl3-2b,internvl3-8b,qwen3vl-4b,qwen3vl-8b \\
      --gpu-id 0

Para apenas regenerar o HTML a partir de resultados já salvos:
  python scripts/analysis/report_vlm_demo.py --html-only
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import transformers
transformers.logging.set_verbosity_error()

import torch
from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = WORKSPACE_ROOT / "results" / "vlm_demo_results.json"
DEFAULT_OUTPUT  = WORKSPACE_ROOT / "results" / "vlm_demo_report.html"

VENV_VLM5 = WORKSPACE_ROOT / ".venv_vlm5" / "bin" / "python"

OCR_PROMPT = (
    "Transcribe all text visible in this document image, preserving the "
    "original layout as closely as possible. Output only the transcribed text."
)

SIMILARITY_PROMPT = """\
You are an AI assistant specialized in document analysis. Your task is to \
compare two company documents and assess their visual similarity based on \
their layout structure.

Analyze the two provided document images and measure their visual similarity based on:
- Shapes and Elements: presence of graphical components, tables, sections, headers.
- Layout Consistency: spatial arrangement of text blocks, margins, and alignments.
- Content Type: similar types of content (tables, forms, paragraphs), regardless of wording.

Scoring: 90-100 Nearly identical · 70-89 Highly similar · 50-69 Moderately similar \
· 30-49 Weak similarity · 0-29 Completely different.

Respond with only a single integer between 0 and 100. No text, no explanation — just the number."""

MODEL_REGISTRY = {
    "internvl3-2b":  "OpenGVLab/InternVL3-2B",
    "internvl3-8b":  "OpenGVLab/InternVL3-8B",
    "internvl3-14b": "OpenGVLab/InternVL3-14B",
    "qwen3vl-2b":    "Qwen/Qwen3-VL-2B-Instruct",
    "qwen3vl-4b":    "Qwen/Qwen3-VL-4B-Instruct",
    "qwen3vl-8b":    "Qwen/Qwen3-VL-8B-Instruct",
    "gemma4-e2b":    "google/gemma-4-E2B-it",
    "gemma4-e4b":    "google/gemma-4-E4B-it",
}
MODELS_VLM5 = {"gemma4-e2b", "gemma4-e4b"}

FAMILY = {
    "internvl3-2b":  "InternVL3", "internvl3-8b":  "InternVL3", "internvl3-14b": "InternVL3",
    "qwen3vl-2b":   "Qwen3-VL",  "qwen3vl-4b":   "Qwen3-VL",  "qwen3vl-8b":   "Qwen3-VL",
    "gemma4-e2b":   "Gemma4",    "gemma4-e4b":   "Gemma4",
}

LABEL = {k: MODEL_REGISTRY[k].split("/")[-1] for k in MODEL_REGISTRY}


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def load_image(path: str, base_dir: str) -> Image.Image:
    return Image.open(Path(base_dir) / path).convert("RGB")


def img_to_b64(img: Image.Image, max_px: int = 400) -> str:
    w, h = img.size
    scale = min(1.0, max_px / max(w, h))
    if scale < 1.0:
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# ---------------------------------------------------------------------------
# Cache cleanup
# ---------------------------------------------------------------------------

def _delete_model_cache(model_id: str) -> None:
    import shutil
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        deleted_mb = 0.0
        for repo in cache_info.repos:
            if repo.repo_id == model_id:
                for revision in repo.revisions:
                    for f in revision.files:
                        p = Path(f.file_path)
                        if p.exists():
                            deleted_mb += p.stat().st_size / 1024 / 1024
                            p.unlink()
                snapshot_dir = Path(repo.repo_path)
                if snapshot_dir.exists():
                    shutil.rmtree(snapshot_dir, ignore_errors=True)
        print(f"  Cache removido: {model_id} (~{deleted_mb:.0f} MB liberados)")
    except Exception as e:
        print(f"  ⚠️  Falha ao remover cache de {model_id}: {e}")


# ---------------------------------------------------------------------------
# BnB config
# ---------------------------------------------------------------------------

def _bnb():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


# ---------------------------------------------------------------------------
# Model adapters
# ---------------------------------------------------------------------------

class _InternVLAdapter:
    def __init__(self, model_id: str, device: str, load_in_4bit: bool = False):
        from transformers import AutoModel, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
            device_map=device,
            quantization_config=_bnb() if load_in_4bit else None,
        ).eval()

    def infer_single(self, img: Image.Image, prompt: str) -> str:
        from cavl_doc.data.transforms import dynamic_preprocess
        from torchvision import transforms
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tiles = dynamic_preprocess(img, image_size=448, use_thumbnail=True, max_num=4)
        pv = torch.stack([tfm(t) for t in tiles]).to(torch.bfloat16).to(
            next(self.model.parameters()).device)
        return self.model.chat(
            self.tokenizer, pv, prompt,
            generation_config={"max_new_tokens": 512, "do_sample": False},
        )

    def infer_pair(self, img_a: Image.Image, img_b: Image.Image, prompt: str) -> str:
        from cavl_doc.data.transforms import dynamic_preprocess
        from torchvision import transforms
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        def proc(img):
            tiles = dynamic_preprocess(img, image_size=448, use_thumbnail=True, max_num=4)
            return torch.stack([tfm(t) for t in tiles])
        pv_a, pv_b = proc(img_a), proc(img_b)
        pv = torch.cat([pv_a, pv_b]).to(torch.bfloat16).to(
            next(self.model.parameters()).device)
        full_prompt = f"Image-1: <image>\nImage-2: <image>\n\n{prompt}"
        return self.model.chat(
            self.tokenizer, pv, full_prompt,
            generation_config={"max_new_tokens": 16, "do_sample": False},
            num_patches_list=[len(pv_a), len(pv_b)],
        )


class _QwenVLAdapter:
    def __init__(self, model_id: str, device: str, load_in_4bit: bool = False):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device,
            quantization_config=_bnb() if load_in_4bit else None,
        ).eval()

    def _generate(self, messages: list, max_new_tokens: int = 512) -> str:
        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise ImportError("pip install qwen-vl-utils")
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        img_inp, vid_inp = process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=img_inp, videos=vid_inp,
            padding=True, return_tensors="pt",
        ).to(next(self.model.parameters()).device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    def infer_single(self, img: Image.Image, prompt: str) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text",  "text": prompt},
        ]}]
        return self._generate(messages, max_new_tokens=512)

    def infer_pair(self, img_a: Image.Image, img_b: Image.Image, prompt: str) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img_a},
            {"type": "image", "image": img_b},
            {"type": "text",  "text": prompt},
        ]}]
        return self._generate(messages, max_new_tokens=16)


class _Gemma4Adapter:
    def __init__(self, model_id: str, device: str, load_in_4bit: bool = False):
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=["vision_tower", "multi_modal_projector",
                                   "language_model.embed_tokens", "lm_head"],
        ) if load_in_4bit else None
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device,
            quantization_config=quant,
        ).eval()

    def _run(self, messages, images, max_new_tokens):
        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, images=images, return_tensors="pt").to(
            next(self.model.parameters()).device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        trimmed = out[0][inputs["input_ids"].shape[-1]:]
        return self.processor.decode(trimmed, skip_special_tokens=True)

    def infer_single(self, img: Image.Image, prompt: str) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        return self._run(messages, [img], 512)

    def infer_pair(self, img_a: Image.Image, img_b: Image.Image, prompt: str) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "image"},
            {"type": "text", "text": prompt},
        ]}]
        return self._run(messages, [img_a, img_b], 16)


def build_adapter(model_key: str, device: str, load_in_4bit: bool = False):
    model_id = MODEL_REGISTRY[model_key]
    print(f"  Carregando {model_id}...")
    kw = dict(model_id=model_id, device=device, load_in_4bit=load_in_4bit)
    if model_key.startswith("internvl"):
        return _InternVLAdapter(**kw)
    if model_key.startswith("qwen"):
        return _QwenVLAdapter(**kw)
    if model_key.startswith("gemma4"):
        return _Gemma4Adapter(**kw)
    raise ValueError(f"Adapter não implementado para {model_key}")


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_model(model_key: str, image_dir: str,
              path_a: str, path_b_same: str, path_b_diff: str,
              device: str, load_in_4bit: bool) -> dict:
    """Roda OCR (img_a) + similaridade (a×b_same, a×b_diff) para um modelo."""
    adapter = build_adapter(model_key, device, load_in_4bit)

    img_a      = load_image(path_a,      image_dir)
    img_b_same = load_image(path_b_same, image_dir)
    img_b_diff = load_image(path_b_diff, image_dir)

    print("    OCR...")
    t0 = time.time()
    ocr_out = adapter.infer_single(img_a, OCR_PROMPT)
    ocr_time = time.time() - t0

    print("    Similaridade (igual)...")
    t0 = time.time()
    sim_same_out = adapter.infer_pair(img_a, img_b_same, SIMILARITY_PROMPT)
    sim_same_time = time.time() - t0

    print("    Similaridade (diferente)...")
    t0 = time.time()
    sim_diff_out = adapter.infer_pair(img_a, img_b_diff, SIMILARITY_PROMPT)
    sim_diff_time = time.time() - t0

    del adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_key":       model_key,
        "model_id":        MODEL_REGISTRY[model_key],
        "family":          FAMILY[model_key],
        "ocr_output":      ocr_out,
        "ocr_time_s":      round(ocr_time, 1),
        "sim_same_output": sim_same_out,
        "sim_same_time_s": round(sim_same_time, 1),
        "sim_diff_output": sim_diff_out,
        "sim_diff_time_s": round(sim_diff_time, 1),
    }


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

FAMILY_COLORS = {
    "InternVL3": "#1565C0",
    "Qwen3-VL":  "#2E7D32",
    "Gemma4":    "#6A1B9A",
}

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f0f2f5; color: #1a1a2e; padding: 28px;
}
h1 { font-size: 1.6rem; margin-bottom: 6px; }
.meta { color: #555; font-size: .9rem; margin-bottom: 32px; }
.family-section { margin-bottom: 40px; }
.family-title {
  font-size: 1.15rem; font-weight: 700; padding: 6px 14px;
  border-radius: 6px; color: #fff; display: inline-block; margin-bottom: 16px;
}
.model-card {
  background: #fff; border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0,0,0,.08);
  margin-bottom: 20px; overflow: hidden;
}
.model-header {
  padding: 12px 20px; font-weight: 600; font-size: 1rem;
  border-bottom: 1px solid #e8e8e8; background: #fafafa;
  display: flex; align-items: center; gap: 10px;
}
.badge {
  font-size: .72rem; font-weight: 600; padding: 2px 8px;
  border-radius: 20px; background: #e3f2fd; color: #1565C0;
}
.tasks { display: grid; grid-template-columns: 1fr 1fr; gap: 0; }
.task {
  padding: 18px 20px; border-right: 1px solid #f0f0f0;
}
.task:last-child { border-right: none; }
.task-title {
  font-size: .78rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: .06em; color: #888; margin-bottom: 12px;
}
.images-row { display: flex; gap: 10px; margin-bottom: 12px; }
.img-box { text-align: center; }
.img-box img {
  max-height: 160px; max-width: 160px; border-radius: 6px;
  border: 1px solid #ddd; object-fit: contain; background: #f9f9f9;
}
.img-label { font-size: .72rem; color: #999; margin-top: 4px; }
.prompt-box {
  background: #f8f9fa; border-left: 3px solid #adb5bd;
  padding: 8px 12px; font-size: .78rem; color: #555;
  font-family: monospace; white-space: pre-wrap;
  border-radius: 0 4px 4px 0; margin-bottom: 10px;
  max-height: 90px; overflow-y: auto;
}
.output-box {
  background: #1e1e2e; color: #cdd6f4;
  padding: 10px 14px; border-radius: 6px;
  font-family: monospace; font-size: .82rem;
  white-space: pre-wrap; min-height: 48px; max-height: 220px;
  overflow-y: auto;
}
.time-tag { font-size: .7rem; color: #aaa; margin-top: 6px; text-align: right; }
.sim-pair { margin-bottom: 18px; }
.sim-pair-label {
  font-size: .72rem; font-weight: 600; color: #fff; padding: 2px 8px;
  border-radius: 4px; display: inline-block; margin-bottom: 8px;
}
.equal-label { background: #2e7d32; }
.diff-label   { background: #c62828; }
"""

def _task_ocr(result: dict, img_a_b64: str) -> str:
    prompt_short = OCR_PROMPT[:180] + ("…" if len(OCR_PROMPT) > 180 else "")
    return f"""
<div class="task">
  <div class="task-title">Tarefa 1 — OCR (imagem única)</div>
  <div class="images-row">
    <div class="img-box">
      <img src="data:image/png;base64,{img_a_b64}" alt="Imagem A">
      <div class="img-label">Imagem A</div>
    </div>
  </div>
  <div class="prompt-box">{prompt_short}</div>
  <div class="output-box">{_esc(result['ocr_output'])}</div>
  <div class="time-tag">⏱ {result['ocr_time_s']}s</div>
</div>"""


def _task_sim(result: dict, img_a_b64: str, img_b_same_b64: str, img_b_diff_b64: str) -> str:
    prompt_short = SIMILARITY_PROMPT[:200] + "…"
    return f"""
<div class="task">
  <div class="task-title">Tarefa 2 — Similaridade (par igual / par diferente)</div>
  <div class="prompt-box">{prompt_short}</div>

  <div class="sim-pair">
    <span class="sim-pair-label equal-label">Par IGUAL</span>
    <div class="images-row">
      <div class="img-box">
        <img src="data:image/png;base64,{img_a_b64}" alt="A">
        <div class="img-label">Imagem A</div>
      </div>
      <div class="img-box">
        <img src="data:image/png;base64,{img_b_same_b64}" alt="B-igual">
        <div class="img-label">Imagem B (igual)</div>
      </div>
    </div>
    <div class="output-box">{_esc(result['sim_same_output'])}</div>
    <div class="time-tag">⏱ {result['sim_same_time_s']}s</div>
  </div>

  <div class="sim-pair">
    <span class="sim-pair-label diff-label">Par DIFERENTE</span>
    <div class="images-row">
      <div class="img-box">
        <img src="data:image/png;base64,{img_a_b64}" alt="A">
        <div class="img-label">Imagem A</div>
      </div>
      <div class="img-box">
        <img src="data:image/png;base64,{img_b_diff_b64}" alt="B-diff">
        <div class="img-label">Imagem B (diferente)</div>
      </div>
    </div>
    <div class="output-box">{_esc(result['sim_diff_output'])}</div>
    <div class="time-tag">⏱ {result['sim_diff_time_s']}s</div>
  </div>
</div>"""


def _esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def generate_html(data: dict, out_path: Path) -> None:
    results   = data["results"]
    img_a_b64      = data["images"]["a"]
    img_b_same_b64 = data["images"]["b_same"]
    img_b_diff_b64 = data["images"]["b_diff"]

    # Group by family
    families: dict[str, list] = {}
    for r in results:
        families.setdefault(r["family"], []).append(r)

    cards_html = []
    for family_name, fam_results in families.items():
        color = FAMILY_COLORS.get(family_name, "#333")
        model_cards = []
        for r in fam_results:
            model_id_label = r["model_id"]
            ocr_html = _task_ocr(r, img_a_b64)
            sim_html = _task_sim(r, img_a_b64, img_b_same_b64, img_b_diff_b64)
            model_cards.append(f"""
<div class="model-card">
  <div class="model-header">
    {r['model_key']}
    <span class="badge">{model_id_label}</span>
  </div>
  <div class="tasks">
    {ocr_html}
    {sim_html}
  </div>
</div>""")

        cards_html.append(f"""
<div class="family-section">
  <div class="family-title" style="background:{color};">{family_name}</div>
  {''.join(model_cards)}
</div>""")

    paths = data.get("paths", {})
    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>VLM Demo — OCR &amp; Similaridade</title>
<style>{CSS}</style>
</head>
<body>
<h1>VLM Demo — OCR &amp; Similaridade de Documentos</h1>
<p class="meta">
  Imagem A: <code>{paths.get('a','')}</code> &nbsp;|&nbsp;
  B igual: <code>{paths.get('b_same','')}</code> &nbsp;|&nbsp;
  B diferente: <code>{paths.get('b_diff','')}</code>
</p>
{''.join(cards_html)}
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"HTML salvo em: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image-dir",    default="/mnt/data/la-cdip/data")
    p.add_argument("--image-a",      default="microbiological_associates_invoice/87103742.tif")
    p.add_argument("--image-b-same", default="microbiological_associates_invoice/87104266.tif")
    p.add_argument("--image-b-diff", default="rjreynolds_costumer_relations/518294700+-4701.tif")
    p.add_argument("--models", default="internvl3-8b,qwen3vl-8b,gemma4-e4b",
                   help="Modelos separados por vírgula")
    p.add_argument("--gpu-id",      type=int, default=0)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--results-json", default=str(DEFAULT_RESULTS))
    p.add_argument("--output-html",  default=str(DEFAULT_OUTPUT))
    p.add_argument("--delete-cache", action="store_true",
                   help="Apaga o cache HuggingFace de cada modelo após a inferência")
    p.add_argument("--html-only",    action="store_true",
                   help="Gera apenas o HTML a partir do JSON existente, sem inferência")
    return p.parse_args()


def main():
    if Path(tempfile.gettempdir()).stat().st_dev != Path("/tmp").stat().st_dev:
        os.environ["TMPDIR"] = "/tmp"
        tempfile.tempdir = "/tmp"

    args = parse_args()
    results_path = Path(args.results_json)
    output_path  = Path(args.output_html)

    if args.html_only:
        data = json.loads(results_path.read_text())
        generate_html(data, output_path)
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    models = [m.strip() for m in args.models.split(",") if m.strip()]

    # Pre-encode images to base64 once
    img_a      = load_image(args.image_a,      args.image_dir)
    img_b_same = load_image(args.image_b_same, args.image_dir)
    img_b_diff = load_image(args.image_b_diff, args.image_dir)

    data = {
        "paths": {
            "a":      args.image_a,
            "b_same": args.image_b_same,
            "b_diff": args.image_b_diff,
        },
        "images": {
            "a":      img_to_b64(img_a),
            "b_same": img_to_b64(img_b_same),
            "b_diff": img_to_b64(img_b_diff),
        },
        "results": [],
    }

    # Load existing results to allow resuming
    if results_path.exists():
        existing = json.loads(results_path.read_text())
        done_keys = {r["model_key"] for r in existing.get("results", [])}
        data["results"] = existing.get("results", [])
        data["images"]  = existing.get("images", data["images"])
        data["paths"]   = existing.get("paths",  data["paths"])
    else:
        done_keys = set()

    for model_key in models:
        if model_key not in MODEL_REGISTRY:
            print(f"[SKIP] {model_key} não reconhecido.")
            continue
        if model_key in done_keys:
            print(f"[SKIP] {model_key} já processado.")
            continue
        if model_key in MODELS_VLM5 and sys.executable != str(VENV_VLM5):
            if not VENV_VLM5.exists():
                print(f"[SKIP] {model_key} requer .venv_vlm5 (não encontrado).")
                continue
            print(f"[INFO] {model_key} requer transformers >= 5.5 → relançando em .venv_vlm5")
            os.execv(str(VENV_VLM5), [str(VENV_VLM5)] + sys.argv)

        print(f"\n{'='*50}\n{model_key}\n{'='*50}")
        try:
            result = run_model(
                model_key, args.image_dir,
                args.image_a, args.image_b_same, args.image_b_diff,
                device, args.load_in_4bit,
            )
            if args.delete_cache:
                _delete_model_cache(MODEL_REGISTRY[model_key])
            data["results"].append(result)
            results_path.parent.mkdir(parents=True, exist_ok=True)
            results_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            print(f"  Salvo: {results_path}")
        except Exception as e:
            print(f"  [ERRO] {model_key}: {e}")

    generate_html(data, output_path)


if __name__ == "__main__":
    main()

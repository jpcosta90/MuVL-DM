#!/usr/bin/env python3
"""
Relatório HTML demonstrando o comportamento de cada VLM em três tarefas,
com três exemplos por tarefa:

  1. OCR       — extração de texto de três documentos individuais
  2. Descrição — descrição visual de três pares de documentos (A e B separadamente)
  3. Métrica   — score de similaridade do paper em três pares (igual e diferente)

Salva inferências em JSON e regenera o HTML a partir dele (--html-only).

Uso:
  python scripts/analysis/report_vlm_demo.py \\
      --image-dir /mnt/data/la-cdip/data \\
      --gpu-id 0

  # Especificar imagens/pares manualmente:
  python scripts/analysis/report_vlm_demo.py \\
      --images-ocr img1.tif,img2.tif,img3.tif \\
      --pairs "a1.tif:b_same1.tif:b_diff1.tif,a2.tif:b_same2.tif:b_diff2.tif,a3.tif:b_same3.tif:b_diff3.tif" \\
      --image-dir /mnt/data/la-cdip/data

  # Só regenerar HTML:
  python scripts/analysis/report_vlm_demo.py --html-only
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import List

import transformers
transformers.logging.set_verbosity_error()

import torch
from PIL import Image

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = WORKSPACE_ROOT / "results" / "vlm_demo_results.json"
DEFAULT_OUTPUT  = WORKSPACE_ROOT / "results" / "vlm_demo_report.html"
VENV_VLM5       = WORKSPACE_ROOT / ".venv_vlm5" / "bin" / "python"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

OCR_PROMPT = (
    "Transcribe all text visible in this document image, preserving the "
    "original layout as closely as possible. Output only the transcribed text."
)

DESCRIPTION_PROMPT = (
    "Describe the visual structure and layout of this document image. "
    "Include: document type, main sections, visual elements present "
    "(tables, forms, headers, paragraphs, signatures, logos, etc.), "
    "and the overall spatial organization of the page."
)

# Idêntico ao eval_vlm_metric.py (incluindo markdown e rubrica completa)
SIMILARITY_PROMPT = """\
Image-1: <image>
Image-2: <image>

You are an AI assistant specialized in document analysis. Your task is to compare two company documents and assess their **visual similarity** based on their layout structure.

**Instructions:**
Analyze the two provided document images and measure their **visual similarity** based on:
- **Shapes and Elements:** Compare the presence of graphical components, tables, sections, headers, and any other visual elements.
- **Layout Consistency:** Evaluate the spatial arrangement of text blocks, margins, and alignments.
- **Content Type:** Ensure that both documents contain similar types of content (e.g., tables, forms, paragraphs), regardless of specific wording.

**Similarity Scoring:**
Assign a **similarity score** between **0 and 100**, where:
- **90-100** → **Nearly identical**: Documents have almost no visual differences.
- **70-89** → **Highly similar**: Documents share the same structure with minor variations (e.g., small alignment changes).
- **50-69** → **Moderately similar**: Key components remain, but there are noticeable structural differences.
- **30-49** → **Weak similarity**: Some elements are shared, but the overall layout is significantly different.
- **0-29** → **Completely different**: The documents do not share a recognizable visual structure.

**Output Format:**
Respond with **only** a single integer between 0 and 100. No text, no explanation, no JSON — just the number."""

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

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
ALL_MODELS  = list(MODEL_REGISTRY)

FAMILY = {
    "internvl3-2b": "InternVL3", "internvl3-8b": "InternVL3", "internvl3-14b": "InternVL3",
    "qwen3vl-2b":   "Qwen3-VL",  "qwen3vl-4b":  "Qwen3-VL",  "qwen3vl-8b":   "Qwen3-VL",
    "gemma4-e2b":   "Gemma4",    "gemma4-e4b":  "Gemma4",
}

# 3 imagens para OCR (task 1)
DEFAULT_IMAGES_OCR = [
    "microbiological_associates_invoice/87103742.tif",
    "cigarrete_portifolio_ad_test/0001223877.tif",
    "document_control_project/0000168142.tif",
]

# 3 pares para descrição (task 2) e métrica (task 3): (a, b_same, b_diff)
DEFAULT_PAIRS = [
    ("microbiological_associates_invoice/87103742.tif",
     "microbiological_associates_invoice/87104266.tif",
     "rjreynolds_costumer_relations/518294700+-4701.tif"),
    ("cigarrete_portifolio_ad_test/0001223877.tif",
     "cigarrete_portifolio_ad_test/0001223905.tif",
     "rjreynolds_costumer_relations/524480861+-0862.tif"),
    ("document_control_project/0000168142.tif",
     "document_control_project/0000169657.tif",
     "analytical_research_division_service_request_and_data_report/2025557028.tif"),
]

# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def load_image(path: str, base_dir: str) -> Image.Image:
    return Image.open(Path(base_dir) / path).convert("RGB")


def img_to_b64(img: Image.Image, max_px: int = 420) -> str:
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
        deleted_mb = 0.0
        for repo in scan_cache_dir().repos:
            if repo.repo_id == model_id:
                for rev in repo.revisions:
                    for f in rev.files:
                        p = Path(f.file_path)
                        if p.exists():
                            deleted_mb += p.stat().st_size / 1024 / 1024
                            p.unlink()
                snap = Path(repo.repo_path)
                if snap.exists():
                    shutil.rmtree(snap, ignore_errors=True)
        print(f"  Cache removido: {model_id} (~{deleted_mb:.0f} MB)")
    except Exception as e:
        print(f"  ⚠️  Falha ao remover cache de {model_id}: {e}")

# ---------------------------------------------------------------------------
# BnB config
# ---------------------------------------------------------------------------

def _bnb():
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
    )

# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------

class _InternVLAdapter:
    def __init__(self, model_id, device, load_in_4bit=False):
        from transformers import AutoModel, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        self.model = AutoModel.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, trust_remote_code=True,
            device_map=device, quantization_config=_bnb() if load_in_4bit else None,
        ).eval()

    def _tiles(self, img):
        from cavl_doc.data.transforms import dynamic_preprocess
        from torchvision import transforms
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        tiles = dynamic_preprocess(img, image_size=448, use_thumbnail=True, max_num=4)
        return torch.stack([tfm(t) for t in tiles])

    def infer_single(self, img, prompt, max_new_tokens=512):
        pv = self._tiles(img).to(torch.bfloat16).to(next(self.model.parameters()).device)
        return self.model.chat(self.tok, pv, prompt,
                               generation_config={"max_new_tokens": max_new_tokens, "do_sample": False})

    def infer_pair(self, img_a, img_b, prompt, max_new_tokens=256):
        pv_a, pv_b = self._tiles(img_a), self._tiles(img_b)
        pv = torch.cat([pv_a, pv_b]).to(torch.bfloat16).to(next(self.model.parameters()).device)
        # Remove image tags from prompt text — passed via pixel_values + num_patches_list
        prompt_text = prompt.replace("Image-1: <image>\nImage-2: <image>\n\n", "")
        return self.model.chat(
            self.tok, pv, f"Image-1: <image>\nImage-2: <image>\n\n{prompt_text}",
            generation_config={"max_new_tokens": max_new_tokens, "do_sample": False},
            num_patches_list=[len(pv_a), len(pv_b)],
        )


class _QwenVLAdapter:
    def __init__(self, model_id, device, load_in_4bit=False):
        from transformers import AutoProcessor, AutoModelForImageTextToText
        self.proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device,
            quantization_config=_bnb() if load_in_4bit else None,
        ).eval()

    def _gen(self, messages, max_new_tokens):
        from qwen_vl_utils import process_vision_info
        text = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(messages)
        inputs = self.proc(text=[text], images=imgs, videos=vids,
                           padding=True, return_tensors="pt").to(next(self.model.parameters()).device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode([o[len(i):] for i, o in zip(inputs.input_ids, out)],
                                      skip_special_tokens=True)[0]

    def infer_single(self, img, prompt, max_new_tokens=512):
        return self._gen([{"role": "user", "content": [
            {"type": "image", "image": img}, {"type": "text", "text": prompt},
        ]}], max_new_tokens)

    def infer_pair(self, img_a, img_b, prompt, max_new_tokens=256):
        prompt_text = prompt.replace("Image-1: <image>\nImage-2: <image>\n\n", "")
        return self._gen([{"role": "user", "content": [
            {"type": "image", "image": img_a}, {"type": "image", "image": img_b},
            {"type": "text", "text": prompt_text},
        ]}], max_new_tokens)


class _Gemma4Adapter:
    def __init__(self, model_id, device, load_in_4bit=False):
        from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
        self.proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        quant = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
            llm_int8_skip_modules=["vision_tower", "multi_modal_projector",
                                   "language_model.embed_tokens", "lm_head"],
        ) if load_in_4bit else None
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map=device,
            quantization_config=quant,
        ).eval()

    def _run(self, messages, images, max_new_tokens):
        text = self.proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.proc(text=text, images=images, return_tensors="pt").to(
            next(self.model.parameters()).device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    def infer_single(self, img, prompt, max_new_tokens=512):
        return self._run(
            [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}],
            [img], max_new_tokens,
        )

    def infer_pair(self, img_a, img_b, prompt, max_new_tokens=256):
        prompt_text = prompt.replace("Image-1: <image>\nImage-2: <image>\n\n", "")
        return self._run(
            [{"role": "user", "content": [
                {"type": "image"}, {"type": "image"}, {"type": "text", "text": prompt_text},
            ]}],
            [img_a, img_b], max_new_tokens,
        )


def build_adapter(model_key, device, load_in_4bit=False):
    model_id = MODEL_REGISTRY[model_key]
    print(f"  Carregando {model_id}...")
    kw = dict(model_id=model_id, device=device, load_in_4bit=load_in_4bit)
    if model_key.startswith("internvl"): return _InternVLAdapter(**kw)
    if model_key.startswith("qwen"):     return _QwenVLAdapter(**kw)
    if model_key.startswith("gemma4"):   return _Gemma4Adapter(**kw)
    raise ValueError(f"Adapter não implementado para {model_key}")

# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def timed(fn, *args, **kw):
    t0 = time.time()
    out = fn(*args, **kw)
    return out, round(time.time() - t0, 1)


def run_model(model_key: str, image_dir: str,
              images_ocr: List[str], pairs: List[tuple],
              device: str, load_in_4bit: bool) -> dict:
    adapter = build_adapter(model_key, device, load_in_4bit)

    # Task 1: OCR em 3 imagens
    ocr_results = []
    for i, path in enumerate(images_ocr):
        print(f"    Tarefa 1 — OCR [{i+1}/3]...")
        img = load_image(path, image_dir)
        out, t = timed(adapter.infer_single, img, OCR_PROMPT)
        ocr_results.append({"output": out, "time_s": t})

    # Tasks 2 e 3: 3 pares
    pair_results = []
    for i, (path_a, path_b_same, path_b_diff) in enumerate(pairs):
        print(f"    Tarefa 2 — Descrição A [{i+1}/3]...")
        img_a      = load_image(path_a,      image_dir)
        img_b_same = load_image(path_b_same, image_dir)
        img_b_diff = load_image(path_b_diff, image_dir)

        desc_a, desc_a_t        = timed(adapter.infer_single, img_a,      DESCRIPTION_PROMPT)
        print(f"    Tarefa 2 — Descrição B_same [{i+1}/3]...")
        desc_b_same, desc_bs_t  = timed(adapter.infer_single, img_b_same, DESCRIPTION_PROMPT)
        print(f"    Tarefa 2 — Descrição B_diff [{i+1}/3]...")
        desc_b_diff, desc_bd_t  = timed(adapter.infer_single, img_b_diff, DESCRIPTION_PROMPT)
        print(f"    Tarefa 3 — Métrica igual [{i+1}/3]...")
        sim_same, sim_same_t    = timed(adapter.infer_pair, img_a, img_b_same, SIMILARITY_PROMPT)
        print(f"    Tarefa 3 — Métrica diferente [{i+1}/3]...")
        sim_diff, sim_diff_t    = timed(adapter.infer_pair, img_a, img_b_diff, SIMILARITY_PROMPT)

        pair_results.append({
            "desc_a_output":      desc_a,     "desc_a_time_s":      desc_a_t,
            "desc_b_same_output": desc_b_same, "desc_b_same_time_s": desc_bs_t,
            "desc_b_diff_output": desc_b_diff, "desc_b_diff_time_s": desc_bd_t,
            "sim_same_output":    sim_same,    "sim_same_time_s":    sim_same_t,
            "sim_diff_output":    sim_diff,    "sim_diff_time_s":    sim_diff_t,
        })

    del adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_key":    model_key,
        "model_id":     MODEL_REGISTRY[model_key],
        "family":       FAMILY[model_key],
        "ocr":          ocr_results,
        "pairs":        pair_results,
    }

# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

FAMILY_COLORS = {"InternVL3": "#1565C0", "Qwen3-VL": "#2E7D32", "Gemma4": "#6A1B9A"}

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f0f2f5; color: #1a1a2e; padding: 24px;
}
h1 { font-size: 1.5rem; margin-bottom: 4px; }
.meta { color: #666; font-size: .82rem; margin-bottom: 28px; line-height: 1.8; }
.meta code { background:#e8e8e8; padding: 1px 5px; border-radius:3px; font-size:.78rem; }
.family-section { margin-bottom: 40px; }
.family-title {
  font-size: 1.05rem; font-weight: 700; padding: 5px 14px;
  border-radius: 6px; color: #fff; display: inline-block; margin-bottom: 12px;
}
.model-card {
  background: #fff; border-radius: 12px;
  box-shadow: 0 2px 8px rgba(0,0,0,.09);
  margin-bottom: 20px; overflow: hidden;
}
.model-header {
  padding: 10px 18px; font-weight: 600; font-size: .95rem;
  border-bottom: 1px solid #ebebeb; background: #fafafa;
  display: flex; align-items: center; gap: 10px;
}
.badge {
  font-size: .68rem; font-weight: 600; padding: 2px 7px;
  border-radius: 20px; background: #e3f2fd; color: #1565C0;
}
/* Três colunas de tarefas */
.tasks { display: grid; grid-template-columns: 1fr 1fr 1fr; border-top: 1px solid #f0f0f0; }
.task { padding: 14px 16px; border-right: 1px solid #f0f0f0; min-width: 0; overflow: hidden; }
.task:last-child { border-right: none; }
.task-header {
  font-size: .7rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: .07em; color: #fff; padding: 3px 10px;
  border-radius: 4px; display: inline-block; margin-bottom: 10px;
}
.task-ocr-header    { background: #455a64; }
.task-desc-header   { background: #00695c; }
.task-metric-header { background: #ad1457; }

/* Exemplos dentro de uma tarefa */
.example { margin-bottom: 14px; padding-bottom: 14px; border-bottom: 1px dashed #eee; }
.example:last-child { border-bottom: none; margin-bottom: 0; padding-bottom: 0; }
.example-label {
  font-size: .66rem; font-weight: 600; color: #aaa;
  text-transform: uppercase; letter-spacing: .05em; margin-bottom: 6px;
}
.images-row { display: flex; gap: 6px; margin-bottom: 7px; flex-wrap: wrap; }
.img-box { text-align: center; }
.img-box img {
  max-height: 120px; max-width: 120px; border-radius: 4px;
  border: 1px solid #ddd; object-fit: contain; background: #f9f9f9;
}
.img-label { font-size: .62rem; color: #bbb; margin-top: 2px; }
.prompt-box {
  background: #f6f7f9; border-left: 3px solid #bbb;
  padding: 5px 9px; font-size: .7rem; color: #777;
  font-family: monospace; white-space: pre-wrap;
  border-radius: 0 3px 3px 0; margin-bottom: 7px;
  max-height: 56px; overflow-y: auto;
}
.output-box {
  background: #1e1e2e; color: #cdd6f4;
  padding: 7px 11px; border-radius: 5px;
  font-family: monospace; font-size: .74rem;
  white-space: pre-wrap; min-height: 36px; max-height: 160px;
  overflow-y: auto;
}
.time-tag { font-size: .62rem; color: #bbb; margin-top: 4px; text-align: right; }
.sim-sub { margin-bottom: 8px; }
.sim-label {
  font-size: .64rem; font-weight: 700; color: #fff;
  padding: 1px 6px; border-radius: 3px; display: inline-block; margin-bottom: 4px;
}
.equal-label { background: #2e7d32; }
.diff-label   { background: #c62828; }
.desc-cols { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 6px; }
.desc-cols > div { min-width: 0; overflow: hidden; }
.desc-col-label { font-size: .62rem; font-weight: 600; color: #999; margin-bottom: 3px; }
"""


def _esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _get(r: dict, key: str) -> str:
    v = r.get(key)
    return _esc(v) if v is not None else "<em style='color:#888'>N/A</em>"


def _get_t(r: dict, key: str) -> str:
    v = r.get(key)
    return f"⏱ {v}s" if v is not None else ""


def _img_tag(b64: str, label: str) -> str:
    return f"""<div class="img-box">
  <img src="data:image/png;base64,{b64}">
  <div class="img-label">{label}</div>
</div>"""


def _task1_ocr(result: dict, imgs_ocr_b64: list, prompt_short: str) -> str:
    examples = []
    for i, (ex, b64) in enumerate(zip(result.get("ocr", []), imgs_ocr_b64)):
        examples.append(f"""<div class="example">
  <div class="example-label">Exemplo {i+1}</div>
  <div class="images-row">{_img_tag(b64, f"Doc {i+1}")}</div>
  <div class="output-box">{_get(ex, 'output')}</div>
  <div class="time-tag">{_get_t(ex, 'time_s')}</div>
</div>""")
    return f"""<div class="task">
  <div class="task-header task-ocr-header">Tarefa 1 — OCR</div>
  <div class="prompt-box">{_esc(prompt_short)}</div>
  {''.join(examples)}
</div>"""


def _task2_desc(result: dict, pairs_b64: list, prompt_short: str) -> str:
    examples = []
    for i, (pr, b64s) in enumerate(zip(result.get("pairs", []), pairs_b64)):
        examples.append(f"""<div class="example">
  <div class="example-label">Par {i+1}</div>
  <div class="desc-cols">
    <div>
      <div class="images-row">{_img_tag(b64s['a'], "Doc A")}</div>
      <div class="desc-col-label">Descrição A</div>
      <div class="output-box">{_get(pr, 'desc_a_output')}</div>
      <div class="time-tag">{_get_t(pr, 'desc_a_time_s')}</div>
    </div>
    <div>
      <div class="images-row">{_img_tag(b64s['b_same'], "Doc B (igual)")}</div>
      <div class="desc-col-label">Descrição B igual</div>
      <div class="output-box">{_get(pr, 'desc_b_same_output')}</div>
      <div class="time-tag">{_get_t(pr, 'desc_b_same_time_s')}</div>
    </div>
    <div>
      <div class="images-row">{_img_tag(b64s['b_diff'], "Doc B (diferente)")}</div>
      <div class="desc-col-label">Descrição B diferente</div>
      <div class="output-box">{_get(pr, 'desc_b_diff_output')}</div>
      <div class="time-tag">{_get_t(pr, 'desc_b_diff_time_s')}</div>
    </div>
  </div>
</div>""")
    return f"""<div class="task">
  <div class="task-header task-desc-header">Tarefa 2 — Descrição Visual</div>
  <div class="prompt-box">{_esc(prompt_short)}</div>
  {''.join(examples)}
</div>"""


def _task3_metric(result: dict, pairs_b64: list, prompt_short: str) -> str:
    examples = []
    for i, (pr, b64s) in enumerate(zip(result.get("pairs", []), pairs_b64)):
        examples.append(f"""<div class="example">
  <div class="example-label">Par {i+1}</div>
  <div class="sim-sub">
    <span class="sim-label equal-label">IGUAL</span>
    <div class="images-row">
      {_img_tag(b64s['a'], "A")}
      {_img_tag(b64s['b_same'], "B")}
    </div>
    <div class="output-box">{_get(pr, 'sim_same_output')}</div>
    <div class="time-tag">{_get_t(pr, 'sim_same_time_s')}</div>
  </div>
  <div class="sim-sub">
    <span class="sim-label diff-label">DIFERENTE</span>
    <div class="images-row">
      {_img_tag(b64s['a'], "A")}
      {_img_tag(b64s['b_diff'], "B")}
    </div>
    <div class="output-box">{_get(pr, 'sim_diff_output')}</div>
    <div class="time-tag">{_get_t(pr, 'sim_diff_time_s')}</div>
  </div>
</div>""")
    return f"""<div class="task">
  <div class="task-header task-metric-header">Tarefa 3 — Métrica do Paper</div>
  <div class="prompt-box">{_esc(prompt_short)}</div>
  {''.join(examples)}
</div>"""


def generate_html(data: dict, out_path: Path) -> None:
    results     = data["results"]
    imgs_ocr_b64 = data["images"]["ocr"]
    pairs_b64    = data["images"]["pairs"]
    paths        = data.get("paths", {})

    ocr_short  = OCR_PROMPT[:120] + "…"
    desc_short = DESCRIPTION_PROMPT[:120] + "…"
    sim_short  = SIMILARITY_PROMPT.replace("Image-1: <image>\nImage-2: <image>\n\n", "")[:180] + "…"

    families: dict[str, list] = {}
    for r in results:
        families.setdefault(r["family"], []).append(r)

    sections = []
    for family_name, fam_results in families.items():
        color = FAMILY_COLORS.get(family_name, "#444")
        cards = []
        for r in fam_results:
            t1 = _task1_ocr(r,    imgs_ocr_b64, ocr_short)
            t2 = _task2_desc(r,   pairs_b64,    desc_short)
            t3 = _task3_metric(r, pairs_b64,    sim_short)
            cards.append(f"""<div class="model-card">
  <div class="model-header">
    {r['model_key']}
    <span class="badge">{r['model_id']}</span>
  </div>
  <div class="tasks">{t1}{t2}{t3}</div>
</div>""")
        sections.append(f"""<div class="family-section">
  <div class="family-title" style="background:{color};">{family_name}</div>
  {''.join(cards)}
</div>""")

    ocr_paths  = paths.get("images_ocr", [])
    pair_paths = paths.get("pairs", [])
    meta_ocr   = " · ".join(f"<code>{p.split('/')[0]}</code>" for p in ocr_paths)
    meta_pairs = "".join(
        f"<br><b>Par {i+1}:</b> <code>{p['a'].split('/')[0]}</code> vs "
        f"<code>{p['b_same'].split('/')[0]}</code> / <code>{p['b_diff'].split('/')[0]}</code>"
        for i, p in enumerate(pair_paths)
    )

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>VLM Demo — OCR · Descrição · Métrica</title>
<style>{CSS}</style>
</head>
<body>
<h1>VLM Demo — OCR · Descrição Visual · Métrica do Paper</h1>
<p class="meta">
  <b>OCR (Tarefa 1):</b> {meta_ocr}
  {meta_pairs}
</p>
{''.join(sections)}
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"HTML salvo em: {out_path}")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_pairs(s: str) -> List[tuple]:
    """Parseia 'a1:b1:c1,a2:b2:c2,a3:b3:c3' em lista de tuplas."""
    result = []
    for item in s.split(","):
        parts = item.strip().split(":")
        if len(parts) != 3:
            raise ValueError(f"Par inválido (esperado a:b_same:b_diff): {item}")
        result.append(tuple(parts))
    return result


def parse_args():
    default_pairs_str = ",".join(f"{a}:{bs}:{bd}" for a, bs, bd in DEFAULT_PAIRS)
    default_ocr_str   = ",".join(DEFAULT_IMAGES_OCR)

    p = argparse.ArgumentParser()
    p.add_argument("--image-dir",   default="/mnt/data/la-cdip/data")
    p.add_argument("--images-ocr",  default=default_ocr_str,
                   help="3 imagens para OCR, separadas por vírgula")
    p.add_argument("--pairs",       default=default_pairs_str,
                   help="3 pares no formato a:b_same:b_diff separados por vírgula")
    p.add_argument("--models",      default=",".join(ALL_MODELS))
    p.add_argument("--gpu-id",      type=int, default=0)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--delete-cache", action="store_true")
    p.add_argument("--results-json", default=str(DEFAULT_RESULTS))
    p.add_argument("--output-html",  default=str(DEFAULT_OUTPUT))
    p.add_argument("--html-only",    action="store_true")
    return p.parse_args()


def main():
    if Path(tempfile.gettempdir()).stat().st_dev != Path("/tmp").stat().st_dev:
        os.environ["TMPDIR"] = "/tmp"
        tempfile.tempdir = "/tmp"

    args = parse_args()
    results_path = Path(args.results_json)
    output_path  = Path(args.output_html)

    if args.html_only:
        generate_html(json.loads(results_path.read_text()), output_path)
        return

    images_ocr = [p.strip() for p in args.images_ocr.split(",")]
    pairs      = parse_pairs(args.pairs)
    models     = [m.strip() for m in args.models.split(",") if m.strip()]

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Pré-carrega imagens como base64
    def b64(path):
        return img_to_b64(load_image(path, args.image_dir))

    data: dict = {
        "paths": {
            "images_ocr": images_ocr,
            "pairs": [{"a": a, "b_same": bs, "b_diff": bd} for a, bs, bd in pairs],
        },
        "images": {
            "ocr":   [b64(p) for p in images_ocr],
            "pairs": [{"a": b64(a), "b_same": b64(bs), "b_diff": b64(bd)}
                      for a, bs, bd in pairs],
        },
        "results": [],
    }

    done_keys: set = set()
    if results_path.exists():
        existing = json.loads(results_path.read_text())
        data["results"] = existing.get("results", [])
        data["images"]  = existing.get("images",  data["images"])
        data["paths"]   = existing.get("paths",   data["paths"])
        done_keys = {r["model_key"] for r in data["results"]}

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
            print(f"[INFO] {model_key} requer .venv_vlm5 → relançando")
            os.execv(str(VENV_VLM5), [str(VENV_VLM5)] + sys.argv)

        print(f"\n{'='*52}\n{model_key}\n{'='*52}")
        try:
            result = run_model(model_key, args.image_dir, images_ocr, pairs,
                               device, args.load_in_4bit)
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

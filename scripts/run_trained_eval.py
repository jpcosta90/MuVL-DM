#!/usr/bin/env python3
"""
Evaluate a trained SiameseInternVL.
Compatible with both "Smart Checkpoints" (new trainer) and Legacy checkpoints.
"""

import os
import argparse
from pathlib import Path
from tqdm import tqdm
import warnings

# Supressão de warnings
warnings.filterwarnings("ignore", message=".*use_reentrant parameter.*")
warnings.filterwarnings("ignore", message=".*None of the inputs have requires_grad.*")

import numpy as np
import torch
from sklearn.metrics import roc_curve
import pandas as pd

# Project imports
from cavl_doc.models.backbone_loader import load_model, warm_up_model
from cavl_doc.models.modeling_cavl import CaVLModel
from cavl_doc.data.dataset import DocumentPairDataset
from cavl_doc.utils.visualization import plot_density

# Tenta importar a função de encode
from cavl_doc.utils.embedding_utils import prepare_inputs_for_multimodal_embedding

def compute_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2
    thr = thresholds[idx]
    return eer, thr

def load_trained_siamese(checkpoint_path, base_model, tokenizer, device, default_proj_out=512):
    print(f"⏳ Carregando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # --- 1. Configuração ---
    if isinstance(checkpoint, dict) and 'config' in checkpoint:
        print("   -> Formato: Smart Checkpoint (Novo)")
        config = checkpoint['config']
        cut_layer = config.get('cut_layer', 27)
        proj_out = config.get('projection_output_dim', default_proj_out)
        hidden_dim = config.get('hidden_dim', 1536)
        if 'metrics' in checkpoint:
            print(f"   -> EER no Treino: {checkpoint['metrics']['eer']*100:.2f}% (Epoch {checkpoint['epoch']})")
    else:
        print("   -> Formato: Legacy (Antigo)")
        cut_layer = 27
        proj_out = default_proj_out
        hidden_dim = 1536

    # --- 2. Detecção de Dimensão ---
    proj_hidden = 4096 
    if isinstance(checkpoint, dict) and 'siam_head' in checkpoint:
        if 'fc1.weight' in checkpoint['siam_head']:
            proj_hidden = checkpoint['siam_head']['fc1.weight'].shape[0]
    elif isinstance(checkpoint, dict) and 'head' in checkpoint:
        if 'fc1.weight' in checkpoint['head']:
            proj_hidden = checkpoint['head']['fc1.weight'].shape[0]
            
    print(f"   -> Config Final: Cut={cut_layer}, Hidden={proj_hidden}, Out={proj_out}")

    # --- 3. Encode Function (Closure) ---
    def _encode_fn(backbone, pv_tensor, cut_layer=cut_layer, **kwargs):
        prompt = "<image> Analyze this document"
        
        # --- CORREÇÃO CRÍTICA: Achatamento de 5D para 4D ---
        # O dataset retorna [1, N_Patches, 3, 448, 448]. O InternVL quer [N_Patches, 3, 448, 448].
        if pv_tensor.dim() == 5:
            b, n, c, h, w = pv_tensor.shape
            # Achata para processamento no vision encoder
            pv_tensor = pv_tensor.view(b * n, c, h, w)
            num_patches = n
        else:
            num_patches = pv_tensor.shape[0] # Assume que dim 0 já são patches ou batch 1

        # A função utilitária lida com a criação dos tokens <IMG_CONTEXT> baseada no num_patches
        inputs = prepare_inputs_for_multimodal_embedding(backbone, tokenizer, pv_tensor, prompt)
        
        out = backbone(
            input_ids=inputs['input_ids'].to(device),
            attention_mask=inputs['attention_mask'].to(device),
            pixel_values=inputs['pixel_values'].to(device, dtype=torch.bfloat16),
            image_flags=inputs['image_flags'].to(device),
            output_hidden_states=True,
            return_dict=True
        )
        hidden_states = out.hidden_states
        lm = backbone.language_model.model
        idx = cut_layer + 1 if len(hidden_states) == (len(lm.layers) + 1) else cut_layer
        return hidden_states[idx], None

    # --- 4. Instancia ---
    siam = CaVLModel(
        backbone=base_model,
        cut_layer=cut_layer,
        hidden_dim=hidden_dim,
        proj_hidden=proj_hidden, 
        proj_out=proj_out,
        encode_fn=_encode_fn
    )
    
    # --- 5. Carrega Pesos ---
    try:
        if isinstance(checkpoint, dict) and 'siam_head' in checkpoint:
            siam.head.load_state_dict(checkpoint['siam_head'])
            siam.pool.load_state_dict(checkpoint['siam_pool'])
            if 'backbone_trainable' in checkpoint:
                print(f"   -> Carregando pesos do backbone ({len(checkpoint['backbone_trainable'])} tensores)...")
                siam.backbone.load_state_dict(checkpoint['backbone_trainable'], strict=False)
        elif isinstance(checkpoint, dict) and 'head' in checkpoint:
            siam.head.load_state_dict(checkpoint['head'])
            if 'pool' in checkpoint: siam.pool.load_state_dict(checkpoint['pool'])
        else:
            siam.head.load_state_dict(checkpoint, strict=False)
            
    except Exception as e:
        print(f"❌ Erro ao carregar pesos: {e}")
        raise e

    siam.to(device)
    siam.eval()
    return siam

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("\nLoading backbone model...")
    backbone, processor, tokenizer, _, _ = load_model(
        model_name=args.model_name,
        load_in_4bit=args.load_in_4bit,
        projection_output_dim=512 # dummy
    )
    backbone.requires_grad_(False)
    backbone.eval()
    warm_up_model(backbone, processor)

    siam = load_trained_siamese(args.checkpoint_path, backbone, tokenizer, device, default_proj_out=args.proj_out)

    dataset = DocumentPairDataset(
        csv_path=args.pairs_csv,
        base_dir=args.base_image_dir,
        input_size=args.input_size,
        max_num=args.max_num_image_tokens,
        device="cpu"
    )
    print(f"Dataset: {len(dataset)} pairs.")

    if args.dataset_name:
        ds_name = args.dataset_name
    else:
        ds_name = Path(args.pairs_csv).parent.name

    ckpt_name = Path(args.checkpoint_path).stem
    results = []

    print("Starting evaluation loop...")
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc=f"Eval {ds_name}"):
            item = dataset[idx]
            # Dataset retorna [Patches, 3, H, W].
            # Precisamos de [1, Patches, 3, H, W] para passar para o encode_fn 
            # que vai detectar o 5D e tratar.
            img_a = item["image_a"].unsqueeze(0).to(device)
            img_b = item["image_b"].unsqueeze(0).to(device)
            label = float(item["label"])

            try:
                # Agora chamamos o modelo de forma que ele use a encode_fn corrigida acima
                # Modo 1: Se o modelo aceita images=...
                z_a = siam(images=img_a)
                z_b = siam(images=img_b)
                
            except Exception as e:
                # Modo 2: Fallback manual chamando a encode_fn diretamente
                # Caso o forward padrão não esteja chamando a encode_fn corretamente
                try:
                    tokens_a, mask_a = siam._extract_tokens_via_encode_fn(img_a, device=device)
                    tokens_b, mask_b = siam._extract_tokens_via_encode_fn(img_b, device=device)
                    
                    z_a = siam.head(siam.pool(tokens_a, mask_a))
                    z_b = siam.head(siam.pool(tokens_b, mask_b))
                except Exception as inner_e:
                    print(f"\n[ERROR] Batch {idx} falhou.")
                    print(f"Input shape: {img_a.shape}")
                    print(f"Erro original: {e}")
                    print(f"Erro fallback: {inner_e}")
                    continue

            if args.metric == "cosine":
                score = torch.nn.functional.cosine_similarity(z_a, z_b).item()
            else:
                score = torch.norm(z_a - z_b, p=2, dim=1).item()

            results.append({"idx": idx, "is_equal": label, "metric_score": score})

    if not results:
        print("Nenhum resultado gerado. Verifique os erros acima.")
        return

    df = pd.DataFrame(results)
    scores = -df["metric_score"].values if args.metric == "euclidean" else df["metric_score"].values
    eer, thr = compute_eer(df["is_equal"].values, scores)

    print(f"\n--- {ds_name} Results ---")
    print(f"EER: {eer*100:.3f}% | Thr: {thr:.4f}")

    if not args.output_csv:
        args.output_csv = f"results/{ds_name}_{ckpt_name}_{args.metric}_eval.csv"
    
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Saved results to: {args.output_csv}")
    
    if args.plot:
        plot_density(df, eer, thr, f"{ckpt_name}_{args.metric}", ds_name, args.metric)
        print("Saved density plot.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-csv", required=True)
    parser.add_argument("--base-image-dir", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--model-name", default="InternVL3-2B")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--metric", default="euclidean", choices=["cosine", "euclidean"])
    parser.add_argument("--input-size", type=int, default=448)
    parser.add_argument("--max-num-image-tokens", type=int, default=12)
    parser.add_argument("--proj-out", type=int, default=512)
    parser.add_argument("--output-csv", type=str)
    parser.add_argument("--plot", action="store_true")
    
    main(parser.parse_args())
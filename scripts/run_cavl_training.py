#!/usr/bin/env python3
# scripts/run_siamese_internvl.py
"""
Script to run the Siamese RL-style fine-tuning loop (calls run_rl_siamese_loop in rl_siamese_trainer.py).

Updated: exposes RL/variance-stabilization and validation hyperparameters.

Usage example (desquantized / fp16 default):
python scripts/run_siamese_internvl.py \
  --dataset-name LA-CDIP \
  --pairs-csv data/LA-CDIP/train_pairs.csv \
  --base-image-dir /path/to/images/ \
  --model-name InternVL3-2B \
  --projection-output-dim 512 \
  --max-num-image-tokens 4 \
  --training-sample-size 2000 \
  --epochs 5 \
  --student-lr 1e-4 \
  --professor-lr 1e-4 \
  --candidate-pool-size 8 \
  --student-batch-size 4 \
  --cut-layer 27 \
  --val-fraction 0.05 \
  --val-min-size 200 \
  --patience 3 \
  --baseline-alpha 0.01 \
  --entropy-coeff 0.01 \
  --lr-reduce-factor 0.5

If you want to load the model in 4-bit quantized mode, pass --load-in-4bit.
"""
import os
import warnings
# --- INÍCIO DO BLOCO "SILENCIADOR NUCLEAR" ---
# Intercepta e mata warnings específicos que ignoram filtros padrão
def _custom_warn(message, category=None, stacklevel=1, source=None):
    msg_str = str(message)
    # Lista de strings parciais para bloquear
    block_list = [
        "use_reentrant parameter should be passed explicitly",
        "None of the inputs have requires_grad=True",
        "torch.utils.checkpoint"
    ]
    if any(s in msg_str for s in block_list):
        return
    # Se não for um dos bloqueados, chama o original
    _original_warn(message, category, stacklevel, source)

# Salva o original e substitui
_original_warn = warnings.warn
warnings.warn = _custom_warn
# --- FIM DO BLOCO ---

# avoid tokenizers parallelism fork warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import time
import random
from pathlib import Path

import torch

# Project imports (adjust if your modules have different names)
from cavl_doc.data.dataset import DocumentPairDataset
from cavl_doc.models.backbone_loader import load_model, warm_up_model
from cavl_doc.utils.helpers import setup_experiment_dir
from cavl_doc.modules.heads import ProjectionHead
from cavl_doc.models.policy import ProfessorNetwork
# Trainer
from cavl_doc.trainers.rl_trainer import run_rl_siamese_loop


def prepare_experiment(args):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    experiment_name = f"{args.dataset_name}_{args.model_name}_siamese_rl_{args.training_sample_size}_{timestamp}"
    outdir = setup_experiment_dir("/mnt/large/checkpoints", experiment_name)
    cfg = {
        'dataset_name': args.dataset_name,
        'model_name': args.model_name,
        'pairs_csv': args.pairs_csv,
        'base_image_dir': args.base_image_dir,
        'projection_output_dim': args.projection_output_dim,
        'max_num_image_tokens': args.max_num_image_tokens,
        'training_sample_size': args.training_sample_size,
        'epochs': args.epochs,
        'student_lr': args.student_lr,
        'professor_lr': args.professor_lr,
        'candidate_pool_size': args.candidate_pool_size,
        'student_batch_size': args.student_batch_size,
        'cut_layer': args.cut_layer,
        'val_fraction': args.val_fraction,
        'val_min_size': args.val_min_size,
        'patience': args.patience,
        'baseline_alpha': args.baseline_alpha,
        'entropy_coeff': args.entropy_coeff,
        'lr_reduce_factor': args.lr_reduce_factor,
        'load_in_4bit': args.load_in_4bit,
        'timestamp': timestamp,
        'outdir': str(outdir)
    }
    with open(os.path.join(outdir, "training_config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"Experiment directory: {outdir}")
    return outdir


def main(args):
    outdir = prepare_experiment(args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 1) load backbone (InternVL) ---
    print(f"Loading backbone model '{args.model_name}' (4-bit={args.load_in_4bit}) ...")
    backbone, processor, tokenizer, _, _ = load_model(
        model_name=args.model_name,
        adapter_path=None,
        load_in_4bit=args.load_in_4bit,
        projection_output_dim=args.projection_output_dim
    )
    backbone.requires_grad_(False)
    warm_up_model(backbone, processor)
    print("Backbone loaded and warmed up (frozen).")


    # default hidden dims — match your trainer expectation or pass via args if different
    LLM_HIDDEN_DIM = 1536
    student_head = ProjectionHead(input_dim=LLM_HIDDEN_DIM, proj_out=args.projection_output_dim).to(device)
    student_head.train()
    print("Student (ProjectionHead) ready.")
    
    professor_model = ProfessorNetwork(input_dim=1).to(device)
    professor_model.train()
    print("Professor network ready.")

    # --- 3) dataset ---
    print("Loading dataset...")
    dataset = DocumentPairDataset(csv_path=args.pairs_csv, base_dir=args.base_image_dir,
                                  input_size=args.input_size, max_num=args.max_num_image_tokens, device='cpu')
    print(f"Dataset loaded with {len(dataset)} samples.")
    if args.training_sample_size > 0 and args.training_sample_size < len(dataset):
        indices = random.sample(range(len(dataset)), args.training_sample_size)
        dataset = torch.utils.data.Subset(dataset, indices)

    # --- 4) call trainer ---
    print("Starting run_rl_siamese_loop trainer ...")
    run_rl_siamese_loop(
        base_model=backbone,
        student_head=student_head,
        professor_model=professor_model,
        tokenizer=tokenizer,
        dataset=dataset,
        epochs=args.epochs,
        student_lr=args.student_lr,
        professor_lr=args.professor_lr,
        device=device,
        output_dir=str(outdir),
        candidate_pool_size=args.candidate_pool_size,
        student_batch_size=args.student_batch_size,
        max_num_image_tokens=args.max_num_image_tokens,
        cut_layer=args.cut_layer,
        val_fraction=args.val_fraction,
        val_min_size=args.val_min_size,
        patience=args.patience,
        lr_reduce_factor=args.lr_reduce_factor,
        baseline_alpha=args.baseline_alpha,
        entropy_coeff=args.entropy_coeff,
        seed=args.seed
    )
    print("run_rl_siamese_loop finished. Checkpoints/logs saved to", outdir)


def parse_args():
    p = argparse.ArgumentParser(description="Script to run Siamese RL training for InternVL.")
    p.add_argument("--model-name", type=str, default="InternVL3-2B")
    p.add_argument("--dataset-name", type=str, default="LA-CDIP")
    p.add_argument("--pairs-csv", type=str, required=True)
    p.add_argument("--base-image-dir", type=str, required=True)
    p.add_argument("--projection-output-dim", type=int, default=512)

    # keep ONLY THIS ONE
    p.add_argument("--max-num-image-tokens", dest="max_num_image_tokens", type=int, default=4)

    p.add_argument("--training-sample-size", dest="training_sample_size", type=int, default=0)
    p.add_argument("--epochs", type=int, default=5)

    p.add_argument("--load-in-4bit", action="store_true", default=False)

    p.add_argument("--student-lr", type=float, default=1e-4)
    p.add_argument("--professor-lr", type=float, default=1e-4)
    p.add_argument("--candidate-pool-size", type=int, default=8)
    p.add_argument("--student-batch-size", type=int, default=4)

    p.add_argument("--cut-layer", type=int, default=27)
    p.add_argument("--input-size", type=int, default=448)

    p.add_argument("--sample-size", dest="training_sample_size", type=int, default=0)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=16)

    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--val-min-size", type=int, default=200)
    p.add_argument("--patience", type=int, default=3)

    p.add_argument("--baseline-alpha", type=float, default=0.01)
    p.add_argument("--entropy-coeff", type=float, default=0.01)
    p.add_argument("--lr-reduce-factor", type=float, default=0.5)

    p.add_argument("--seed", type=int, default=42)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)

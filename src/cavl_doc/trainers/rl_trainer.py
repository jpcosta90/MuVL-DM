# src/cavl_doc/finetuning/rl_siamese_trainer.py
import os
import csv
import logging
import math
from tqdm import tqdm
import random
import time

import numpy as np
from sklearn.metrics import roc_curve

import torch
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Subset

# Project imports
from cavl_doc.modules.losses import ContrastiveLoss

# prepare_inputs_for_multimodal_embedding may be in different modules
from cavl_doc.utils.embedding_utils import prepare_inputs_for_multimodal_embedding

# siam builder
from cavl_doc.models.modeling_cavl import build_cavl_model

logger = logging.getLogger(__name__)
EMBEDDING_PROMPT = "<image> Analyze this document"

# --- Helper para Média Móvel ---
class AverageMeter:
    """Computa e armazena a média e o valor atual"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def rl_full_collate_fn(batch):
    """
    Collate function consistent with DocumentPairDataset returning lists.
    """
    img_a_list = [item['image_a'] for item in batch]
    img_b_list = [item['image_b'] for item in batch]
    labels = torch.tensor([item['label'] for item in batch], dtype=torch.float32)
    return img_a_list, img_b_list, labels

# ----------------- Validation / EER helpers -----------------
def pairwise_eer_from_scores(labels: np.ndarray, scores: np.ndarray):
    if len(labels) == 0:
        return 1.0, 0.0
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    idx = np.argmin(abs_diffs)
    eer = (fpr[idx] + fnr[idx]) / 2.0
    thr = thresholds[idx]
    return eer, thr

def validate_siam_on_loader(siam, val_loader, device, student_criterion):
    siam.eval()
    all_labels = []
    all_scores = []
    losses = []
    with torch.no_grad():
        for img_a_list, img_b_list, labels in val_loader:
            emb_a_list = []
            emb_b_list = []
            for i in range(len(img_a_list)):
                pv_a = img_a_list[i]
                pv_b = img_b_list[i]
                za = siam(images=pv_a.to(device))
                zb = siam(images=pv_b.to(device))
                emb_a_list.append(za)
                emb_b_list.append(zb)
            za = torch.cat(emb_a_list, dim=0)
            zb = torch.cat(emb_b_list, dim=0)
            labels = labels.to(device)
            ind_losses = student_criterion.forward_individual(za, zb, labels)
            losses.append(ind_losses.mean().item())
            scores = torch.nn.functional.cosine_similarity(za, zb, dim=-1).cpu().numpy()
            all_scores.append(scores)
            all_labels.append(labels.cpu().numpy())

    if len(losses) == 0:
        return float('nan'), 1.0, 0.0
    all_scores = np.concatenate(all_scores, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    mean_loss = float(np.mean(losses))
    eer, thr = pairwise_eer_from_scores(all_labels, all_scores)
    return mean_loss, eer, thr

# ----------------- Main Trainer -----------------
def run_rl_siamese_loop(
    base_model,
    student_head,
    professor_model,
    tokenizer,
    dataset,
    epochs,
    student_lr,
    professor_lr,
    device,
    output_dir,
    candidate_pool_size,
    student_batch_size,
    max_num_image_tokens,
    cut_layer=27,
    projection_output_dim=512, # Passado pelo novo script
    val_fraction=0.05,
    val_min_size=200,
    patience=3,
    lr_reduce_factor=0.5,
    baseline_alpha=0.01,
    entropy_coeff=0.01,
    seed=42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(f"Iniciando loop de RL (Siamese tail+head) — K={candidate_pool_size}, B={student_batch_size}")
    device = torch.device(device if isinstance(device, str) else device)

    # Configuração crítica para salvar
    model_config = {
        'cut_layer': cut_layer,
        'projection_output_dim': projection_output_dim,
        'max_num_image_tokens': max_num_image_tokens,
        'hidden_dim': 1536
    }

    def _encode_fn(backbone, pv_tensor, cut_layer=cut_layer, **kwargs):
        inputs = prepare_inputs_for_multimodal_embedding(backbone, tokenizer, pv_tensor, EMBEDDING_PROMPT)
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

    # Build wrapper
    siam = build_cavl_model(
        backbone=base_model, 
        cut_layer=cut_layer, 
        encode_fn=_encode_fn,
        # Usa as dimensões do head passado se possível
        pool_dim=student_head.ln.normalized_shape[0] if hasattr(student_head, 'ln') else 1536,
        proj_hidden=getattr(student_head, 'fc1').out_features if hasattr(student_head, 'fc1') else 4096,
        proj_out=projection_output_dim,
        set_trainable=True
    )
    siam.to(device)
    siam.set_default_trainable()
    
    print("Trainable parameters in Siamese wrapper:")
    siam.trainable_summary()

    professor_model.to(device)
    professor_model.train()

    # Optimizers
    trainable_params = [p for n,p in siam.named_parameters() if p.requires_grad]
    student_optimizer = optim.Adam(trainable_params, lr=student_lr * lr_reduce_factor)
    professor_optimizer = optim.Adam(professor_model.parameters(), lr=professor_lr * lr_reduce_factor)
    student_criterion = ContrastiveLoss(margin=1.0).to(device)

    # Splits
    if isinstance(dataset, Subset): full_ds, ds_indices = dataset.dataset, list(dataset.indices)
    else: full_ds, ds_indices = dataset, list(range(len(dataset)))
    
    val_size = min(max(val_min_size, int(len(ds_indices) * val_fraction)), len(ds_indices)//10)
    if val_size <= 0: val_size = min(val_min_size, len(ds_indices)//10)
    
    val_indices = ds_indices[-val_size:]
    train_indices = ds_indices[:-val_size]
    if len(train_indices) == 0:
        random.shuffle(ds_indices)
        val_indices = ds_indices[:val_size]
        train_indices = ds_indices[val_size:]

    train_loader = DataLoader(Subset(full_ds, train_indices), batch_size=candidate_pool_size, shuffle=True, num_workers=4, collate_fn=rl_full_collate_fn)
    val_loader = DataLoader(Subset(full_ds, val_indices), batch_size=16, shuffle=False, num_workers=2, collate_fn=rl_full_collate_fn)

    # CSV Logger
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, "training_log.csv")
    log_file = open(log_file_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch', 'batch', 'aluno_loss', 'prof_loss', 'reward', 'baseline', 'entropy', 'adv_std', 'val_mean_loss', 'val_eer'])
    print(f" -> Log: {log_file_path}")

    best_val_eer = 1.0
    no_improve = 0
    global_batch_step = 0
    baseline = 0.0

    # Forward helper
    def student_forward_pass(pv_list_a, pv_list_b, train_student=True):
        if train_student: siam.train()
        else: siam.eval()
        emb_a_list, emb_b_list = [], []
        for pv in pv_list_a:
            if train_student: z = siam(images=pv.to(device))
            else:
                with torch.no_grad(): z = siam(images=pv.to(device))
            emb_a_list.append(z)
        for pv in pv_list_b:
            if train_student: z = siam(images=pv.to(device))
            else:
                with torch.no_grad(): z = siam(images=pv.to(device))
            emb_b_list.append(z)
        return torch.cat(emb_a_list, dim=0), torch.cat(emb_b_list, dim=0)

    print("Iniciando treinamento...")
    for epoch in range(epochs):
        print(f"\n--- Época {epoch + 1}/{epochs} ---")
        
        # --- Reset Medidores da Época ---
        avg_student_loss = AverageMeter()
        avg_reward = AverageMeter()
        avg_prof_loss = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Época {epoch+1}", unit="batch")

        for batch_idx, (img_a_list, img_b_list, labels) in enumerate(pbar):
            labels = labels.to(device).float()

            # 1. Professor Turn
            professor_model.train()
            with torch.no_grad():
                emb_a_all, emb_b_all = student_forward_pass(img_a_list, img_b_list, train_student=False)
                state_losses = student_criterion.forward_individual(emb_a_all, emb_b_all, labels.to(device))
                denom = (state_losses.max() - state_losses.min()).item()
                denom = denom if denom != 0 else 1.0
                state_losses_norm = (state_losses - state_losses.min()) / (denom + 1e-6)
                state_input = state_losses_norm.unsqueeze(-1)

            action_logits = professor_model(state_input).squeeze(-1)
            prob_dist = Categorical(logits=action_logits)
            selected_indices = prob_dist.sample((student_batch_size,))
            selected_log_probs = prob_dist.log_prob(selected_indices)

            # 2. Student Turn
            student_indices = selected_indices.tolist()
            student_img_a = [img_a_list[i] for i in student_indices]
            student_img_b = [img_b_list[i] for i in student_indices]
            student_labels = labels[student_indices]

            student_optimizer.zero_grad()
            student_emb_a, student_emb_b = student_forward_pass(student_img_a, student_img_b, train_student=True)
            student_loss = student_criterion(student_emb_a, student_emb_b, student_labels.to(device))
            student_loss.backward()
            student_optimizer.step()

            # 3. Professor Update
            with torch.no_grad():
                student_individual = student_criterion.forward_individual(student_emb_a.detach(), student_emb_b.detach(), student_labels.to(device))
            
            rewards = student_individual.detach()
            current_avg_reward = float(rewards.mean().item())
            
            advantage = rewards - baseline
            adv_std = advantage.std(unbiased=False).clamp(min=1e-6)
            advantage_norm = (advantage - advantage.mean()) / (adv_std + 1e-6)

            professor_optimizer.zero_grad()
            prof_loss_val = - (selected_log_probs * advantage_norm).mean()
            entropy = prob_dist.entropy().mean()
            prof_loss = prof_loss_val - entropy_coeff * entropy
            prof_loss.backward()
            professor_optimizer.step()

            baseline = (1 - baseline_alpha) * baseline + baseline_alpha * current_avg_reward

            # --- Log Updates ---
            global_batch_step += 1
            
            # Atualiza médias móveis
            avg_student_loss.update(student_loss.item())
            avg_reward.update(current_avg_reward)
            avg_prof_loss.update(prof_loss.item())

            # Pbar mostra as MÉDIAS
            pbar.set_postfix({
                'S_Loss': f"{avg_student_loss.avg:.4f}",
                'Rew': f"{avg_reward.avg:.4f}",
                'Best_EER': f"{best_val_eer:.3f}"
            })

            log_writer.writerow([
                epoch + 1, global_batch_step,
                f"{student_loss.item():.6f}", f"{prof_loss.item():.6f}",
                f"{current_avg_reward:.6f}", f"{baseline:.6f}", f"{entropy.item():.6f}",
                f"{adv_std.item():.6f}", "", ""
            ])

        pbar.close()

        # Validação
        val_mean_loss, val_eer, val_thr = validate_siam_on_loader(siam, val_loader, device, student_criterion)
        print(f"Validation — mean_loss: {val_mean_loss:.6f}, EER: {100*val_eer:.3f}%")
        
        metrics = {'eer': val_eer, 'loss': val_mean_loss, 'threshold': val_thr}
        log_writer.writerow([epoch + 1, "epoch_end", "", "", "", "", "", "", f"{val_mean_loss:.6f}", f"{val_eer:.6f}"])

        # Save Smart Checkpoint
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            no_improve = 0
            # Salva usando o método interno da classe se existir, senão fallback manual
            if hasattr(siam, 'save_smart'):
                 # Se você implementou save_smart no modelo, use aqui. 
                 # Mas como não implementamos isso no SiameseInternVL antigo,
                 # vamos salvar manualmente usando o dicionário robusto
                 pass
            
            # Monta o Smart Checkpoint Manualmente (Compatível com seu script de Eval)
            backbone_trainable = {n: p.detach().cpu() for n, p in siam.backbone.named_parameters() if p.requires_grad}
            ckpt = {
                'epoch': epoch,
                'metrics': metrics,
                'config': model_config,
                'siam_pool': siam.pool.state_dict(),
                'siam_head': siam.head.state_dict(),
                'backbone_trainable': backbone_trainable,
                'professor_state': professor_model.state_dict()
            }
            torch.save(ckpt, os.path.join(output_dir, "best_siam.pt"))
            print("[CHECKPOINT] Saved new best siam.")
        else:
            no_improve += 1
            print(f"[INFO] No improvement ({no_improve}/{patience}).")

        if no_improve >= patience:
            print("Early stopping.")
            break

    log_file.close()
    print("\n✅ Treinamento concluído.")
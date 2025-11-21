# src/finetuning/rl_siamese_trainer.py
import os
import warnings

# --- BLOCO DE SUPRESSÃO DE WARNINGS (Deve ficar no topo) ---
# 1. Ignora o aviso de 'use_reentrant' (mudança futura do PyTorch)
warnings.filterwarnings("ignore", category=UserWarning, message=".*use_reentrant.*")
# 2. Ignora o aviso de 'requires_grad' (comum quando congelamos o backbone)
warnings.filterwarnings("ignore", category=UserWarning, message=".*None of the inputs have requires_grad.*")
# 3. Filtra qualquer UserWarning vindo diretamente do módulo de checkpoint para garantir
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")

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
from src.models.professor import ProfessorNetwork
from src.finetuning.losses import ContrastiveLoss
from src.models.siamese_internVL import SiameseInternVL

logger = logging.getLogger(__name__)

# --- 2. Helper para Média Móvel ---
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
    Collate function consistent with DocumentPairDataset returning lists of tensors.
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
    """
    Runs siam.eval() on val_loader.
    Handles dynamic image sizes by processing one by one if necessary.
    """
    siam.eval()
    all_labels = []
    all_scores = []
    losses = []
    
    with torch.no_grad():
        for img_a_list, img_b_list, labels in val_loader:
            emb_a_list = []
            emb_b_list = []
            
            # Processa lista (necessário se as imagens tiverem shapes dinâmicos/diferentes números de patches)
            for i in range(len(img_a_list)):
                # Adiciona dimensão de batch [1, C, H, W]
                pv_a = img_a_list[i].unsqueeze(0).to(device)
                pv_b = img_b_list[i].unsqueeze(0).to(device)
                
                za = siam(images=pv_a)
                zb = siam(images=pv_b)
                
                emb_a_list.append(za)
                emb_b_list.append(zb)
            
            za = torch.cat(emb_a_list, dim=0)  # (B, D)
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

# ----------------- Smart Checkpoint Helper -----------------
def save_smart_checkpoint(path, siam_model, professor_model, epoch, metrics, model_config):
    """
    Salva um checkpoint unificado com configuração e apenas pesos treináveis.
    """
    # Extrai apenas os pesos do backbone que foram treinados (ex: camadas LoRA ou unfreezed)
    backbone_trainable = {
        n: p.detach().cpu() for n, p in siam_model.backbone.named_parameters() if p.requires_grad
    }
    
    checkpoint = {
        'epoch': epoch,
        'metrics': metrics,          # {'eer': ..., 'loss': ...}
        'config': model_config,      # Dicionário com cut_layer, dims, etc.
        'siam_pool': siam_model.pool.state_dict(),
        'siam_head': siam_model.head.state_dict(),
        'backbone_trainable': backbone_trainable,
        'professor_state': professor_model.state_dict()
    }
    torch.save(checkpoint, path)

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
    candidate_pool_size, # K
    student_batch_size,  # B
    max_num_image_tokens,
    cut_layer = 27,
    projection_output_dim = 512, # Add to config
    val_fraction = 0.05,
    val_min_size = 200,
    patience = 3,
    lr_reduce_factor = 0.5,
    baseline_alpha = 0.01,
    entropy_coeff = 0.01,
    seed = 42
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    print(f"Iniciando loop de RL (Siamese tail+head) — K={candidate_pool_size}, B={student_batch_size}")
    device = torch.device(device if isinstance(device, str) else device)

    # Configuração crítica para salvar e carregar depois
    model_config = {
        'cut_layer': cut_layer,
        'projection_output_dim': projection_output_dim,
        'max_num_image_tokens': max_num_image_tokens,
        'hidden_dim': 1536 # InternVL default
    }

    # ---------------- build siamese wrapper ----------------
    # Agora usamos a classe diretamente, sem funções auxiliares complexas
    siam = SiameseInternVL(
        backbone=base_model,
        tokenizer=tokenizer,
        cut_layer=cut_layer,
        head=student_head, # Usa o head passado (ou cria novo se None)
        proj_out=projection_output_dim,
        prompt="<image> Analyze this document"
    )
    siam.to(device)
    
    # Configura parâmetros treináveis (congela backbone, solta camada de corte)
    siam.set_default_trainable()
    print("Trainable parameters in Siamese wrapper:")
    siam.trainable_summary()

    professor_model.to(device)
    professor_model.train()

    # ------- optimizers -------
    trainable_params = [p for n,p in siam.named_parameters() if p.requires_grad]
    if len(trainable_params) == 0:
        raise RuntimeError("No trainable parameters detected. Check set_default_trainable().")
        
    student_optimizer = optim.Adam(trainable_params, lr=student_lr * lr_reduce_factor)
    professor_optimizer = optim.Adam(professor_model.parameters(), lr=professor_lr * lr_reduce_factor)
    student_criterion = ContrastiveLoss(margin=1.0).to(device)

    # ---------------- create train/val splits ----------------
    if isinstance(dataset, Subset):
        full_ds = dataset.dataset
        ds_indices = list(dataset.indices)
    else:
        full_ds = dataset
        ds_indices = list(range(len(full_ds)))

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

    # CSV logger
    os.makedirs(output_dir, exist_ok=True)
    log_file_path = os.path.join(output_dir, "training_log.csv")
    log_file = open(log_file_path, 'w', newline='')
    log_writer = csv.writer(log_file)
    log_writer.writerow(['epoch', 'batch', 'aluno_loss', 'prof_loss', 'reward', 'baseline', 'entropy', 'adv_std', 'val_mean_loss', 'val_eer'])

    best_val_eer = 1.0
    no_improve = 0
    global_batch_step = 0
    baseline = 0.0 

    # Helper: Forward seguro para listas de tensores
    def student_forward_pass(pv_list_a, pv_list_b, train_student=True):
        if train_student: siam.train()
        else: siam.eval()

        emb_a_list = []
        emb_b_list = []
        
        # Loop manual para garantir suporte a imagens de tamanhos variados (comum no InternVL)
        for pv in pv_list_a:
            pv = pv.unsqueeze(0).to(device)
            z = siam(images=pv) if train_student else siam(images=pv)
            if not train_student: z = z.detach()
            emb_a_list.append(z)
            
        for pv in pv_list_b:
            pv = pv.unsqueeze(0).to(device)
            z = siam(images=pv) if train_student else siam(images=pv)
            if not train_student: z = z.detach()
            emb_b_list.append(z)

        return torch.cat(emb_a_list, dim=0), torch.cat(emb_b_list, dim=0)

    print("Iniciando o loop de co-treinamento (Siamese)...")
    
    for epoch in range(epochs):
        print(f"\n--- Época {epoch + 1}/{epochs} ---")
        
        # --- Inicializa medidores para a época ---
        avg_student_loss = AverageMeter()
        avg_prof_loss = AverageMeter()
        avg_reward = AverageMeter()

        pbar = tqdm(train_loader, desc=f"Época {epoch+1}", unit="batch")

        for batch_idx, (img_a_list, img_b_list, labels) in enumerate(pbar):
            labels = labels.to(device).float()

            # --- 1) Professor turn ---
            professor_model.train()
            with torch.no_grad():
                emb_a_all, emb_b_all = student_forward_pass(img_a_list, img_b_list, train_student=False)
                state_losses = student_criterion.forward_individual(emb_a_all, emb_b_all, labels)
                
                denom = (state_losses.max() - state_losses.min()).item()
                denom = denom if denom != 0 else 1.0
                state_losses_norm = (state_losses - state_losses.min()) / (denom + 1e-6)
                state_input = state_losses_norm.unsqueeze(-1) # (K,1)

            action_logits = professor_model(state_input).squeeze(-1)
            prob_dist = Categorical(logits=action_logits)
            selected_indices = prob_dist.sample((student_batch_size,))
            selected_log_probs = prob_dist.log_prob(selected_indices)

            # --- 2) Student turn ---
            student_indices = selected_indices.tolist()
            student_img_a = [img_a_list[i] for i in student_indices]
            student_img_b = [img_b_list[i] for i in student_indices]
            student_labels = labels[student_indices]

            student_optimizer.zero_grad()
            student_emb_a, student_emb_b = student_forward_pass(student_img_a, student_img_b, train_student=True)
            student_loss = student_criterion(student_emb_a, student_emb_b, student_labels)
            student_loss.backward()
            student_optimizer.step()

            # --- 3) Professor update ---
            with torch.no_grad():
                student_individual = student_criterion.forward_individual(student_emb_a.detach(), student_emb_b.detach(), student_labels)
            
            rewards = student_individual.detach()
            current_avg_reward = float(rewards.mean().item())
            
            # Advantage calc
            advantage = rewards - baseline
            adv_std = advantage.std(unbiased=False).clamp(min=1e-6)
            advantage_norm = (advantage - advantage.mean()) / (adv_std + 1e-6)

            professor_optimizer.zero_grad()
            prof_loss = - (selected_log_probs * advantage_norm).mean()
            entropy = prob_dist.entropy().mean()
            total_prof_loss = prof_loss - entropy_coeff * entropy
            total_prof_loss.backward()
            professor_optimizer.step()

            baseline = (1 - baseline_alpha) * baseline + baseline_alpha * current_avg_reward

            # --- Updates de Log ---
            global_batch_step += 1
            
            # Atualiza as médias móveis
            avg_student_loss.update(student_loss.item())
            avg_prof_loss.update(total_prof_loss.item())
            avg_reward.update(current_avg_reward)

            # Barra de progresso mostra as MÉDIAS, não o valor instável do batch
            pbar.set_postfix({
                'S_Loss': f"{avg_student_loss.avg:.4f}",
                'Rew': f"{avg_reward.avg:.4f}",
                'Best_EER': f"{best_val_eer:.3f}"
            })

            log_writer.writerow([
                epoch + 1, global_batch_step,
                f"{student_loss.item():.6f}", f"{total_prof_loss.item():.6f}",
                f"{current_avg_reward:.6f}", f"{baseline:.6f}", f"{entropy.item():.6f}",
                f"{adv_std.item():.6f}", "", ""
            ])

        pbar.close()

        # ---------- Validation ----------
        val_mean_loss, val_eer, val_thr = validate_siam_on_loader(siam, val_loader, device, student_criterion)
        print(f"Validation — Loss: {val_mean_loss:.6f}, EER: {100*val_eer:.3f}%")
        
        log_writer.writerow([epoch + 1, "end", "", "", "", "", "", "", f"{val_mean_loss:.6f}", f"{val_eer:.6f}"])
        
        metrics = {'eer': val_eer, 'loss': val_mean_loss, 'threshold': val_thr}

        # Save Best (Smart Checkpoint)
        if val_eer < best_val_eer:
            best_val_eer = val_eer
            no_improve = 0
            save_smart_checkpoint(
                os.path.join(output_dir, "best_siam.pt"),
                siam, professor_model, epoch, metrics, model_config
            )
            print("[CHECKPOINT] Saved best_siam.pt (New Best EER)")
        else:
            no_improve += 1
            print(f"[INFO] No improve ({no_improve}/{patience})")

        if no_improve >= patience:
            print("Early stopping.")
            break

    log_file.close()
    print("\n✅ Treinamento concluído.")
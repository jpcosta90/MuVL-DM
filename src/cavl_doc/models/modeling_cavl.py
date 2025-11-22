# src/models/siamese_internVL.py
from typing import Any, Callable, Optional, List
import torch
import torch.nn as nn

from cavl_doc.modules.heads import ProjectionHead
from cavl_doc.modules.poolers import AttentionPooling
# ----------------------
# AttentionPooling + ProjectionHead (Mantidos idênticos)
# ----------------------
# ----------------------
# Siamese wrapper (Ajustado: Warnings + Loader Inteligente)
# ----------------------
class CaVLModel(nn.Module):
    def __init__(self,
                 backbone: Any,
                 cut_layer: int = 27,
                 hidden_dim: int = 1536,
                 proj_hidden: int = 4096,
                 proj_out: int = 512,
                 num_pool_heads: int = 8,
                 encode_fn: Optional[Callable] = None):
        super().__init__()
        self.backbone = backbone
        
        # --- [NOVO] Correção Técnica para Warnings e Checkpointing ---
        # Isso evita o "use_reentrant" warning e o "None of the inputs have requires_grad"
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
        if hasattr(self.backbone, "enable_input_require_grads"):
            self.backbone.enable_input_require_grads()
        else:
            # Fallback seguro
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            self.backbone.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
        # -------------------------------------------------------------

        self.cut_layer = cut_layer
        self.encode_fn = encode_fn
        self.hidden_dim = hidden_dim

        # Limpeza do backbone (opcional, mantido do seu código)
        try:
            if hasattr(self.backbone.language_model, "lm_head"):
                self.backbone.language_model.lm_head = nn.Identity()
            if hasattr(self.backbone.language_model.model, "norm"):
                self.backbone.language_model.model.norm = nn.Identity()
        except Exception:
            pass

        self.pool = AttentionPooling(hidden_dim, num_heads=num_pool_heads)
        self.head = ProjectionHead(input_dim=hidden_dim, proj_hidden=proj_hidden, proj_out=proj_out)

        self.freeze_all_backbone()

    # --- [NOVO] Método de Carregamento Inteligente ---
    def load_smart_checkpoint(self, path: str, map_location='cpu'):
        """
        Carrega checkpoints complexos (Head + Pool + Backbone Parcial) gerados pelo trainer novo.
        """
        print(f"Carregando pesos de: {path}")
        ckpt = torch.load(path, map_location=map_location)
        
        # 1. Verifica se é o formato Smart Checkpoint (dicionário com chaves específicas)
        if isinstance(ckpt, dict) and 'siam_head' in ckpt:
            print(" -> Formato Smart Checkpoint detectado.")
            self.head.load_state_dict(ckpt['siam_head'])
            self.pool.load_state_dict(ckpt['siam_pool'])
            
            if 'backbone_trainable' in ckpt:
                print(f" -> Carregando {len(ckpt['backbone_trainable'])} parâmetros treinados no backbone...")
                # strict=False é essencial aqui pois carregamos apenas parciais
                self.backbone.load_state_dict(ckpt['backbone_trainable'], strict=False)
        
        # 2. Fallback para formato antigo (apenas dict do head/pool)
        elif isinstance(ckpt, dict) and 'head' in ckpt:
            print(" -> Formato Legacy (head/pool) detectado.")
            self.head.load_state_dict(ckpt['head'])
            if 'pool' in ckpt:
                self.pool.load_state_dict(ckpt['pool'])
                
        else:
            print(" -> Aviso: Formato desconhecido, tentando carregar direto no Head.")
            self.head.load_state_dict(ckpt, strict=False)

    # --- Métodos Originais (Mantidos para não quebrar o treino atual) ---
    def freeze_all_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_params_by_substrings(self, substrings: List[str]):
        skipped = []
        unfrozen = []
        for name, p in self.backbone.named_parameters():
            for s in substrings:
                if s in name:
                    if not getattr(p, "dtype", None) or not p.dtype.is_floating_point:
                        skipped.append((name, str(p.dtype)))
                        break
                    p.requires_grad = True
                    unfrozen.append(name)
                    break
        if skipped:
            print(f"[WARN] Skipped non-float params: {len(skipped)}")
        if unfrozen:
            print(f"[INFO] Unfroze {len(unfrozen)} parameter tensors.")

    def set_default_trainable(self):
        self.freeze_all_backbone()
        cut = self.cut_layer
        keys = [
            f"layers.{cut}.self_attn",
            f"layers.{cut}.mlp",
            f"layers.{cut}.input_layernorm",
            f"layers.{cut}.post_attention_layernorm",
        ]
        self.unfreeze_params_by_substrings(keys)

    def trainable_summary(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
        return total, trainable

    def _extract_tokens_via_encode_fn(self, images: torch.Tensor, device: Optional[torch.device] = None, **encode_kwargs):
        assert callable(self.encode_fn), "encode_fn not provided."
        out = self.encode_fn(self.backbone, images, cut_layer=self.cut_layer, **(encode_kwargs or {}))
        if isinstance(out, tuple):
            return out[0], out[1]
        return out, None

    def _extract_tokens_via_hidden_states(self, input_ids=None, attention_mask=None, device=None, **kwargs):
        lm = self.backbone.language_model.model
        call_args = dict(output_hidden_states=True, return_dict=True)
        if input_ids is not None: call_args['input_ids'] = input_ids.to(next(self.parameters()).device)
        if attention_mask is not None: call_args['attention_mask'] = attention_mask.to(next(self.parameters()).device)
        call_args.update(kwargs)
        out = lm(**call_args)
        hidden_states = out.hidden_states
        idx = self.cut_layer + 1 if len(hidden_states) == (len(lm.layers) + 1) else self.cut_layer
        return hidden_states[idx], None

    def forward(self,
                images: Optional[torch.Tensor] = None,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                device: Optional[torch.device] = None,
                encode_kwargs: Optional[dict] = None) -> torch.Tensor:
        device = device or (next(self.parameters()).device)
        if self.encode_fn is not None and images is not None:
            tokens, mask = self._extract_tokens_via_encode_fn(images.to(device), device=device, **(encode_kwargs or {}))
        else:
            tokens, mask = self._extract_tokens_via_hidden_states(input_ids=input_ids, attention_mask=attention_mask, device=device, **(encode_kwargs or {}))
        
        pooled = self.pool(tokens, mask=mask)
        z = self.head(pooled)
        return z

    # Helpers legados de save/load (opcional manter)
    def save_head(self, path: str):
        torch.save({'pool': self.pool.state_dict(), 'head': self.head.state_dict()}, path)
    def load_head(self, path: str, map_location=None):
        sd = torch.load(path, map_location=map_location)
        self.pool.load_state_dict(sd['pool'])
        self.head.load_state_dict(sd['head'])

# ----------------------
# Factory (Mantida compatível)
# ----------------------
def build_siamese_cavl(
        backbone: Any,
        cut_layer: int = 27,
        encode_fn: Optional[Callable] = None,
        hidden_dim: int = 1536,
        proj_hidden: int = 4096,
        proj_out: int = 512,
        num_pool_heads: int = 8,
        pool_dim: Optional[int] = None,
        set_trainable: bool = True,
        **kwargs # Ignora args extras como tokenizer se não usados
) -> CaVLModel:
    if pool_dim is not None: hidden_dim = pool_dim

    siam = CaVLModel(
        backbone=backbone,
        cut_layer=cut_layer,
        hidden_dim=hidden_dim,
        proj_hidden=proj_hidden,
        proj_out=proj_out,
        num_pool_heads=num_pool_heads,
        encode_fn=encode_fn
    )
    if set_trainable:
        siam.set_default_trainable()
    return siam
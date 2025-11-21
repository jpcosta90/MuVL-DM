# src/models/siamese_internVL.py
from typing import Any, Callable, Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------
# AttentionPooling + ProjectionHead (Mantidos do seu código original)
# ----------------------
class AttentionPooling(nn.Module):
    """
    Attention-based pooling that learns a global query vector and attends over token sequence.
    Input: tokens (B, seq_len, hidden_dim)
    Output: pooled (B, hidden_dim)
    """
    def __init__(self, hidden_dim: int = 1536, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        # batch_first=False API (seq_len, batch, embed)
        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=False)
        # learnable query token (1, hidden_dim)
        self.query = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        self.ln = nn.LayerNorm(hidden_dim, eps=1e-6)

    def forward(self, tokens: torch.Tensor, mask: Optional[torch.Tensor] = None):
        if tokens is None:
            raise ValueError("tokens must be provided to AttentionPooling")

        b, seq_len, d = tokens.shape

        # Ensure dtype consistency
        target_dtype = self.query.dtype
        if tokens.dtype != target_dtype:
            tokens = tokens.to(dtype=target_dtype)

        tokens_t = tokens.transpose(0, 1)  # (seq_len, B, d)
        q = self.query.unsqueeze(1).expand(1, b, d)  # (1, B, d)

        if mask is not None:
            if mask.dtype == torch.bool:
                key_padding_mask = ~mask
            else:
                key_padding_mask = ~(mask.bool())
        else:
            key_padding_mask = None

        attn_out, _ = self.mha(q, tokens_t, tokens_t, key_padding_mask=key_padding_mask)
        pooled = attn_out.squeeze(0)  # (B, d)
        pooled = self.ln(pooled)
        return pooled


class ProjectionHead(nn.Module):
    """
    Projection MLP head: input_dim -> proj_hidden -> proj_out
    """
    def __init__(self, input_dim: int = 1536, proj_hidden: int = 4096, proj_out: int = 512, use_norm: bool = True):
        super().__init__()
        self.ln = nn.LayerNorm(input_dim, eps=1e-6)
        self.fc1 = nn.Linear(input_dim, proj_hidden, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(proj_hidden, proj_out, bias=True)
        self.use_norm = use_norm

        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        if self.fc1.bias is not None: nn.init.zeros_(self.fc1.bias)
        if self.fc2.bias is not None: nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor):
        x = self.ln(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        if self.use_norm:
            x = F.normalize(x, p=2, dim=-1)
        return x


# ----------------------
# Siamese wrapper (Atualizado para ser Self-Contained)
# ----------------------
class SiameseInternVL(nn.Module):
    """
    Wrapper autônomo para InternVLChatModel.
    Gerencia tokenização interna, extração na cut_layer, pooling e projeção.
    """
    def __init__(self,
                 backbone: Any,
                 tokenizer: Any,
                 cut_layer: int = 27,
                 prompt: str = "<image> Analyze this document",
                 # Head configs (se não passar head pronto)
                 head: Optional[nn.Module] = None,
                 hidden_dim: int = 1536,
                 proj_hidden: int = 4096,
                 proj_out: int = 512,
                 num_pool_heads: int = 8):
        super().__init__()
        self.backbone = backbone

        # --- CORREÇÃO DOS WARNINGS AQUI ---
        
        # 1. Resolve o warning "use_reentrant parameter should be passed explicitly"
        # Forçamos a configuração correta no backbone
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        # 2. Resolve o warning "None of the inputs have requires_grad=True"
        # Quando usamos checkpointing em modelo congelado, a entrada PRECISA ter gradiente
        # senão o checkpointing quebra a cadeia. Essa função do HF faz isso automaticamente.
        if hasattr(self.backbone, "enable_input_require_grads"):
            self.backbone.enable_input_require_grads()
        else:
            # Fallback manual caso o backbone não tenha o método (raro em HF modernos)
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            self.backbone.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # ----------------------------------
        
        self.tokenizer = tokenizer
        self.cut_layer = cut_layer
        self.prompt = prompt
        self.hidden_dim = hidden_dim

        # 1. Limpeza do Backbone (remover cabeças de LM originais para economizar memória/evitar bugs)
        try:
            if hasattr(self.backbone.language_model, "lm_head"):
                self.backbone.language_model.lm_head = nn.Identity()
            if hasattr(self.backbone.language_model.model, "norm"):
                self.backbone.language_model.model.norm = nn.Identity()
        except Exception:
            pass

        # 2. Componentes de Pooling e Head
        self.pool = AttentionPooling(hidden_dim, num_heads=num_pool_heads)
        
        if head is not None:
            self.head = head
        else:
            self.head = ProjectionHead(input_dim=hidden_dim, proj_hidden=proj_hidden, proj_out=proj_out)

        # 3. Congelar backbone por padrão
        self.freeze_all_backbone()

    # --- Internal Input Prep (A mágica da compatibilidade) ---
    def _prepare_inputs(self, pixel_values: torch.Tensor):
        """
        Gera input_ids e attention_mask a partir do prompt armazenado e anexa as imagens.
        """
        device = pixel_values.device
        batch_size = pixel_values.shape[0]
        
        # Tokenização do prompt fixo
        text_inputs = self.tokenizer(
            [self.prompt] * batch_size,
            return_tensors='pt',
            padding=True,
            max_length=80, # Margem de segurança
            truncation=True
        ).to(device)

        return {
            'input_ids': text_inputs.input_ids,
            'attention_mask': text_inputs.attention_mask,
            # Força bfloat16 se o backbone for bf16 (comum em InternVL)
            'pixel_values': pixel_values.to(dtype=torch.bfloat16), 
            'image_flags': torch.ones(batch_size, 1).to(device)
        }

    # --- Forward Single (Imagem -> Embedding) ---
    def forward_single(self, images: torch.Tensor):
        """
        Processa um batch de imagens e retorna embeddings normalizados.
        images: (B, 3, H, W)
        """
        inputs = self._prepare_inputs(images)
        
        # Passa pelo Backbone
        outputs = self.backbone(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            pixel_values=inputs['pixel_values'],
            image_flags=inputs['image_flags'],
            output_hidden_states=True,
            return_dict=True
        )
        
        # Seleção da camada (Cut Layer)
        hidden_states = outputs.hidden_states
        # Ajuste de índice: hidden_states geralmente tem (embeddings + layers)
        # Se len for 33 (embed + 32 layers), e queremos a layer 27 (index 28)
        target_idx = self.cut_layer + 1 if len(hidden_states) > self.cut_layer + 1 else self.cut_layer
        
        tokens = hidden_states[target_idx] # (B, Seq_Len, Hidden_Dim)
        
        # Pooling e Projeção
        pooled = self.pool(tokens) # (B, Hidden_Dim)
        z = self.head(pooled)      # (B, Proj_Out)
        
        return z

    # --- Forward Flexível (Treino ou Inferência) ---
    def forward(self, 
                images: Optional[torch.Tensor] = None, 
                image_a: Optional[torch.Tensor] = None, 
                image_b: Optional[torch.Tensor] = None):
        """
        Modos:
        1. image_a + image_b -> Retorna (emb_a, emb_b) para treino siamês.
        2. images -> Retorna emb para inferência.
        """
        if image_a is not None and image_b is not None:
            emb_a = self.forward_single(image_a)
            emb_b = self.forward_single(image_b)
            return emb_a, emb_b
        
        if images is not None:
            return self.forward_single(images)
            
        raise ValueError("Forneça 'images' (inferência) ou o par 'image_a' e 'image_b' (treino).")

    # --- Utilitários de Treinamento (Mantidos para compatibilidade com Trainer) ---
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
            print(f"[INFO] Unfroze {len(unfrozen)} params in backbone.")

    def set_default_trainable(self):
        self.freeze_all_backbone()
        cut = self.cut_layer
        # Unfreeze typical LoRA/Adapter targets or Full Finetuning targets at cut_layer
        keys = [
            f"layers.{cut}.self_attn",
            f"layers.{cut}.mlp",
            f"layers.{cut}.input_layernorm",
            f"layers.{cut}.post_attention_layernorm",
        ]
        self.unfreeze_params_by_substrings(keys)

    def trainable_summary(self):
        total = 0
        trainable = 0
        print("Trainable parameter summary:")
        for n, p in self.named_parameters():
            nparams = p.numel()
            total += nparams
            if p.requires_grad:
                trainable += nparams
        pct = 100.0 * trainable / total if total > 0 else 0.0
        print(f"Total params: {total:,} | Trainable: {trainable:,} ({pct:.2f}%)")
        return total, trainable

# ----------------------
# Factory (Atualizada)
# ----------------------
def build_siamese_internvl(
        backbone: Any,
        tokenizer: Any, # <--- Novo argumento obrigatório
        cut_layer: int = 27,
        hidden_dim: int = 1536,
        proj_hidden: int = 4096,
        proj_out: int = 512,
        num_pool_heads: int = 8,
        prompt: str = "<image> Analyze this document",
        head: Optional[nn.Module] = None,
        set_trainable: bool = True,
        **kwargs # Captura args legados como encode_fn e pool_dim
) -> SiameseInternVL:
    
    # Compatibilidade com chamadas antigas que usavam 'pool_dim'
    if 'pool_dim' in kwargs and kwargs['pool_dim'] is not None:
        hidden_dim = kwargs['pool_dim']

    siam = SiameseInternVL(
        backbone=backbone,
        tokenizer=tokenizer,
        cut_layer=cut_layer,
        hidden_dim=hidden_dim,
        proj_hidden=proj_hidden,
        proj_out=proj_out,
        num_pool_heads=num_pool_heads,
        prompt=prompt,
        head=head
    )
    
    if set_trainable:
        siam.set_default_trainable()
        
    return siam
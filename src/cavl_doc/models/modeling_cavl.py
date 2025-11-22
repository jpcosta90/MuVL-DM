# src/cavl_doc/models/modeling_cavl.py
from typing import Any, Callable, Optional, Tuple, List
import torch
import torch.nn as nn
from transformers import PreTrainedModel, AutoModel

# Imports internos
from cavl_doc.models.configuration_cavl import CaVLConfig
from cavl_doc.modules.poolers import AttentionPooling
from cavl_doc.modules.heads import ProjectionHead

class CaVLModel(PreTrainedModel):
    config_class = CaVLConfig
    base_model_prefix = "backbone" 

    def __init__(self, 
                 backbone_or_config: Any, 
                 # Argumentos opcionais para modo legado (treino)
                 cut_layer: int = 27,
                 hidden_dim: int = 1536,
                 proj_hidden: int = 4096,
                 proj_out: int = 512,
                 num_pool_heads: int = 8,
                 encode_fn: Optional[Callable] = None,
                 head: Optional[nn.Module] = None,
                 pooler: Optional[nn.Module] = None,
                 tokenizer: Any = None,
                 prompt: str = "<image> Analyze this document"):
        
        # --- Lógica Híbrida de Inicialização ---
        if isinstance(backbone_or_config, CaVLConfig):
            # MODO 1: Inicialização via Config (from_pretrained)
            config = backbone_or_config
            super().__init__(config)
            
            # Carrega o backbone do HF automaticamente
            self.backbone = AutoModel.from_pretrained(
                config.backbone_name, 
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
            )
            # Sobrescreve parâmetros com os da config
            self.cut_layer = config.cut_layer
            self.hidden_dim = config.hidden_dim
            # Configs não usadas na inferência direta
            self.encode_fn = None 
            self.tokenizer = None # Assume que o usuário tokenizou antes ou usará processor
            
        else:
            # MODO 2: Inicialização Manual (Seu Treino Atual)
            # Cria uma config on-the-fly para manter o PreTrainedModel feliz
            config = CaVLConfig(
                cut_layer=cut_layer,
                hidden_dim=hidden_dim,
                proj_hidden=proj_hidden,
                proj_out=proj_out,
                num_pool_heads=num_pool_heads
            )
            super().__init__(config)
            self.backbone = backbone_or_config # Aqui é o objeto backbone passado
            self.cut_layer = cut_layer
            self.encode_fn = encode_fn
            self.tokenizer = tokenizer # Armazena se passado (compatibilidade)

        self.prompt = prompt

        # --- Configurações Comuns ---
        
        # Correção de Warnings / Checkpointing
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
        if hasattr(self.backbone, "enable_input_require_grads"):
            self.backbone.enable_input_require_grads()
        else:
            def mk_grad(m, i, o): o.requires_grad_(True)
            self.backbone.get_input_embeddings().register_forward_hook(mk_grad)

        # Limpeza
        try:
            if hasattr(self.backbone.language_model, "lm_head"): self.backbone.language_model.lm_head = nn.Identity()
            if hasattr(self.backbone.language_model.model, "norm"): self.backbone.language_model.model.norm = nn.Identity()
        except: pass

        # Montagem dos Módulos (Usa config ou argumentos passados)
        dim = config.hidden_dim
        ph = config.proj_hidden
        po = config.proj_out
        nph = config.num_pool_heads

        if pooler is not None:
            self.pool = pooler
        else:
            self.pool = AttentionPooling(dim, num_heads=nph)

        if head is not None:
            self.head = head
        else:
            self.head = ProjectionHead(dim, ph, po)

        self.freeze_all_backbone()

    # --- Utilitários ---
    def freeze_all_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad = False

    def set_default_trainable(self):
        self.freeze_all_backbone()
        cut = self.config.cut_layer # Usa da config
        keys = [f"layers.{cut}.self_attn", f"layers.{cut}.mlp", f"layers.{cut}.input_layernorm", f"layers.{cut}.post_attention_layernorm"]
        for n, p in self.backbone.named_parameters():
            for k in keys:
                if k in n and p.dtype.is_floating_point:
                    p.requires_grad = True
                    break

    def trainable_summary(self):
        tot = sum(p.numel() for p in self.parameters())
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Trainable: {tr:,} / {tot:,} ({100*tr/tot:.2f}%)")
        return tot, tr

    # --- Forward Logic (Híbrida) ---
    def _extract_tokens_via_encode_fn(self, images, device=None, **encode_kwargs):
        assert callable(self.encode_fn), "encode_fn not provided in legacy mode."
        out = self.encode_fn(self.backbone, images, cut_layer=self.cut_layer, **encode_kwargs)
        return (out[0], out[1]) if isinstance(out, tuple) else (out, None)

    def _extract_tokens_via_hidden_states(self, input_ids=None, attention_mask=None, device=None, **kwargs):
        lm = self.backbone.language_model.model
        call_args = dict(output_hidden_states=True, return_dict=True)
        if input_ids is not None: call_args['input_ids'] = input_ids.to(next(self.parameters()).device)
        if attention_mask is not None: call_args['attention_mask'] = attention_mask.to(next(self.parameters()).device)
        call_args.update(kwargs)
        out = lm(**call_args)
        hs = out.hidden_states
        idx = self.config.cut_layer + 1 if len(hs) == (len(lm.layers) + 1) else self.config.cut_layer
        return hs[idx], None

    def forward(self, images=None, input_ids=None, attention_mask=None, device=None, encode_kwargs=None, image_a=None, image_b=None):
        device = device or (next(self.parameters()).device)
        
        # Modo Siamese Training (Legado)
        if image_a is not None and image_b is not None:
            za = self.forward(images=image_a, device=device)
            zb = self.forward(images=image_b, device=device)
            return za, zb

        # Modo Legacy (com encode_fn injetada)
        if self.encode_fn is not None and images is not None:
            tokens, mask = self._extract_tokens_via_encode_fn(images.to(device), device=device, **(encode_kwargs or {}))
        else:
            # Modo HF Padrão (espera input_ids processados ou implementa lógica interna no futuro)
            tokens, mask = self._extract_tokens_via_hidden_states(input_ids=input_ids, attention_mask=attention_mask, device=device, **(encode_kwargs or {}))
        
        pooled = self.pool(tokens, mask=mask)
        return self.head(pooled)

# ----------------------
# Factory (MANTIDA IGUAL AO QUE VOCÊ USA)
# ----------------------
def build_cavl_model(
        backbone: Any,
        cut_layer: int = 27,
        encode_fn: Optional[Callable] = None,
        hidden_dim: int = 1536,
        proj_hidden: int = 4096,
        proj_out: int = 512,
        num_pool_heads: int = 8,
        pool_dim: Optional[int] = None,
        set_trainable: bool = True,
        tokenizer: Any = None, # Aceita para não quebrar, passa para o modelo
        **kwargs 
) -> CaVLModel:
    if pool_dim is not None: hidden_dim = pool_dim

    # A chamada permanece idêntica, passando os argumentos explicitamente.
    # O CaVLModel vai cair no "MODO 2" do __init__ e funcionar como antes.
    model = CaVLModel(
        backbone_or_config=backbone, # Passamos o backbone no 1º argumento
        cut_layer=cut_layer,
        hidden_dim=hidden_dim,
        proj_hidden=proj_hidden,
        proj_out=proj_out,
        num_pool_heads=num_pool_heads,
        encode_fn=encode_fn,
        tokenizer=tokenizer
    )
    if set_trainable:
        model.set_default_trainable()
    return model
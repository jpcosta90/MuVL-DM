# src/cavl_doc/models/configuration_cavl.py
from transformers import PretrainedConfig

class CaVLConfig(PretrainedConfig):
    model_type = "cavl"

    def __init__(
        self,
        backbone_name="OpenGVLab/InternVL3-2B",
        cut_layer=27,
        hidden_dim=1536,
        proj_hidden=4096,
        proj_out=512,
        num_pool_heads=8,
        use_gradient_checkpointing=True,
        force_input_grads=True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.backbone_name = backbone_name
        self.cut_layer = cut_layer
        self.hidden_dim = hidden_dim
        self.proj_hidden = proj_hidden
        self.proj_out = proj_out
        self.num_pool_heads = num_pool_heads
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.force_input_grads = force_input_grads
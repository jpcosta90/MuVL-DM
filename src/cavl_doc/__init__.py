__version__ = "0.1.0"

# Atalhos diretos para as classes principais (Ouro!)
from .models.configuration_cavl import CaVLConfig
from .models.modeling_cavl import CaVLModel
from .models.backbone_loader import load_model

# Se quiser expor submodulos
from . import data
from . import utils
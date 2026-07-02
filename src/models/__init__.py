from src.models.llava import LLaVAModel
from src.models.minicpm import MiniCPMModel
from src.models.qwen2_5_vl import Qwen25VLModel

_REGISTRY = {
    'llava':       LLaVAModel,
    'minicpm':     MiniCPMModel,
    'qwen2_5_vl':  Qwen25VLModel,
}


def get_vlm(config):
    name = config['model']['name'].lower()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](config)

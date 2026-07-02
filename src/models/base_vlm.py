from abc import ABC, abstractmethod


class BaseVLM(ABC):
    """
    All VLM wrappers must implement this interface.

    Public API used by evaluate.py:
      vlm.load()
      vlm.generate_caption(image_path)   ← dispatches to the active strategy

    To add a new strategy (e.g. few_shot), implement _few_shot() in the subclass
    and add it to the dispatch map in generate_caption().
    """

    def __init__(self, config):
        self.config = config
        self.model = None
        self.processor = None

    @abstractmethod
    def load(self):
        """Download and initialise model + processor. Called once before the eval loop."""
        pass

    # ── Strategy dispatch ─────────────────────────────────────────────────────

    def generate_caption(self, image_path: str) -> str:
        strategy = self.config['model'].get('strategy', 'zero_shot')
        if strategy == 'zero_shot':
            return self._zero_shot(image_path)
        elif strategy == 'few_shot':
            return self._few_shot(image_path)
        raise ValueError(f"Unknown strategy '{strategy}'. Available: zero_shot, few_shot")

    @abstractmethod
    def _zero_shot(self, image_path: str) -> str:
        """Generate a caption with no examples — only the prompt and the image."""
        pass

    def _few_shot(self, image_path: str) -> str:
        """Override in subclass to support in-context example prompting."""
        raise NotImplementedError(
            f"{self.__class__.__name__} has not implemented few_shot strategy yet."
        )

    # ── MLflow experiment name ────────────────────────────────────────────────

    def get_model_id(self) -> str:
        """
        DNA string used as the MLflow experiment name.
        Shows exactly what was run so experiments are self-describing.

        Example: VLM_qwen2_5_vl_Qwen-Qwen2.5-VL-7B-Instruct_300samples_zeroshot_8bit
        """
        name     = self.config['model']['name']
        hf_id    = self.config['model'].get('hf_model_id', name).replace('/', '-')
        samples  = self.config['evaluation']['num_samples']
        strategy = self.config['model'].get('strategy', 'zero_shot')
        quant    = "8bit" if self.config['model'].get('load_in_8bit') else "fp16"
        return f"VLM_{name}_{hf_id}_{samples}samples_{strategy}_{quant}"

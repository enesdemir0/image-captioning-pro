import torch
from PIL import Image
from src.models.base_vlm import BaseVLM


class MiniCPMModel(BaseVLM):
    """MiniCPM-V-8B — openbmb/MiniCPM-V-2_6"""

    def load(self):
        from transformers import AutoModel, AutoTokenizer

        hf_id = self.config['model']['hf_model_id']
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)

        # NOTE: MiniCPM-V-2_6's trust_remote_code model class predates the
        # `all_tied_weights_keys` API that recent transformers' bnb 8-bit
        # quantizer uses to auto-detect skip modules, so requesting 8-bit
        # here crashes with AttributeError during from_pretrained(). Load
        # in fp16 instead — LLaVA/Qwen2.5-VL are unaffected since they're
        # natively integrated rather than trust_remote_code.
        load_kwargs = {
            "trust_remote_code": True,
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }

        self.model = AutoModel.from_pretrained(hf_id, **load_kwargs)
        self.model.eval()
        print(f"MiniCPM-V loaded: {hf_id}")

    def _zero_shot(self, image_path: str) -> str:
        prompt_text = self.config['model'].get('prompt', 'Describe this image in one sentence.')
        image = Image.open(image_path).convert("RGB")
        msgs = [{'role': 'user', 'content': [image, prompt_text]}]
        result = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer)
        return result.strip()

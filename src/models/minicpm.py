from PIL import Image
from src.models.base_vlm import BaseVLM


class MiniCPMModel(BaseVLM):
    """MiniCPM-V-8B — openbmb/MiniCPM-V-2_6"""

    def load(self):
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

        hf_id = self.config['model']['hf_model_id']
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)

        load_kwargs = {"trust_remote_code": True, "device_map": "auto"}
        if self.config['model'].get('load_in_8bit', True):
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self.model = AutoModel.from_pretrained(hf_id, **load_kwargs)
        self.model.eval()
        print(f"MiniCPM-V loaded: {hf_id}")

    def _zero_shot(self, image_path: str) -> str:
        prompt_text = self.config['model'].get('prompt', 'Describe this image in one sentence.')
        image = Image.open(image_path).convert("RGB")
        msgs = [{'role': 'user', 'content': [image, prompt_text]}]
        result = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer)
        return result.strip()

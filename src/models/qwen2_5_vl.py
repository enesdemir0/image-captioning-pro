import torch
from PIL import Image
from src.models.base_vlm import BaseVLM


class Qwen25VLModel(BaseVLM):
    """Qwen2.5-VL-7B — Qwen/Qwen2.5-VL-7B-Instruct"""

    def load(self):
        from transformers import AutoProcessor, BitsAndBytesConfig

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
        except ImportError:
            from transformers import Qwen2VLForConditionalGeneration as ModelClass

        hf_id = self.config['model']['hf_model_id']
        self.processor = AutoProcessor.from_pretrained(hf_id)

        load_kwargs = {"device_map": "auto"}
        if self.config['model'].get('load_in_8bit', True):
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self.model = ModelClass.from_pretrained(hf_id, **load_kwargs)
        print(f"Qwen2.5-VL loaded: {hf_id}")

    def _zero_shot(self, image_path: str) -> str:
        prompt_text = self.config['model'].get('prompt', 'Describe this image in one sentence.')
        image = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text}
            ]
        }]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=[image], return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.config['model'].get('max_new_tokens', 50)
            )
        generated = output[0][inputs.input_ids.shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

import torch
from PIL import Image
from src.models.base_vlm import BaseVLM


class Qwen25VLModel(BaseVLM):
    """Qwen2.5-VL-7B — Qwen/Qwen2.5-VL-7B-Instruct"""

    def load(self):
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        hf_id = self.config['model']['hf_model_id']
        self.processor = AutoProcessor.from_pretrained(hf_id)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            hf_id,
            load_in_8bit=self.config['model'].get('load_in_8bit', True),
            device_map="auto"
        )
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

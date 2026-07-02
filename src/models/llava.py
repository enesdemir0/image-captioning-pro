import torch
from PIL import Image
from src.models.base_vlm import BaseVLM


class LLaVAModel(BaseVLM):
    """LLaVA-7B — llava-hf/llava-1.5-7b-hf"""

    def load(self):
        from transformers import LlavaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

        hf_id = self.config['model']['hf_model_id']
        self.processor = AutoProcessor.from_pretrained(hf_id)

        load_kwargs = {"device_map": "auto"}
        if self.config['model'].get('load_in_8bit', True):
            load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        else:
            load_kwargs["torch_dtype"] = torch.float16

        self.model = LlavaForConditionalGeneration.from_pretrained(hf_id, **load_kwargs)
        print(f"LLaVA loaded: {hf_id}")

    def _zero_shot(self, image_path: str) -> str:
        prompt_text = self.config['model'].get('prompt', 'Describe this image in one sentence.')
        prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.config['model'].get('max_new_tokens', 50)
            )
        full = self.processor.decode(output[0], skip_special_tokens=True)
        return full.split("ASSISTANT:")[-1].strip()

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
        # `all_tied_weights_keys` API that recent transformers uses both in
        # the bnb 8-bit quantizer AND in accelerate's infer_auto_device_map
        # (triggered by device_map="auto"), so either one crashes with
        # AttributeError during from_pretrained(). Load plain and place it
        # on the device manually instead — LLaVA/Qwen2.5-VL are unaffected
        # since they're natively integrated rather than trust_remote_code.
        self.model = AutoModel.from_pretrained(
            hf_id, trust_remote_code=True, dtype=torch.float16
        )
        self.model = self.model.to(self.config['model'].get('device', 'cuda'))
        self.model.eval()
        print(f"MiniCPM-V loaded: {hf_id}")

    def _zero_shot(self, image_path: str) -> str:
        prompt_text = self.config['model'].get('prompt', 'Describe this image in one sentence.')
        image = Image.open(image_path).convert("RGB")
        msgs = [{'role': 'user', 'content': [image, prompt_text]}]
        result = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer)
        return result.strip()

import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image


class ViTGPT2Captioner:
    """Pre-trained ViT + GPT-2 captioning model via HuggingFace Transformers.

    Wraps nlpconnect/vit-gpt2-image-captioning so it fits the same interface
    pattern as the custom CNN+RNN stack: instantiate once, call generate_caption.
    """

    _MODEL_ID = "nlpconnect/vit-gpt2-image-captioning"

    def __init__(self):
        self.model             = VisionEncoderDecoderModel.from_pretrained(self._MODEL_ID)
        self.feature_extractor = ViTImageProcessor.from_pretrained(self._MODEL_ID)
        self.tokenizer         = AutoTokenizer.from_pretrained(self._MODEL_ID)
        self.device            = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def generate_caption(
        self,
        image: Image.Image,
        max_length: int   = 16,
        num_beams: int    = 4,
        temperature: float = 1.0,
        top_k: int        = 50,
        top_p: float      = 0.9,
    ) -> str:
        pixel_values = self.feature_extractor(
            images=image, return_tensors="pt"
        ).pixel_values.to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                pixel_values,
                max_length=max_length,
                num_beams=num_beams,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                return_dict_in_generate=True,
            ).sequences

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

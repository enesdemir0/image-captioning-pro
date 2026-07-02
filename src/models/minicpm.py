import torch
from PIL import Image
from src.models.base_vlm import BaseVLM


class MiniCPMModel(BaseVLM):
    """MiniCPM-V-8B — openbmb/MiniCPM-V-2_6"""

    def load(self):
        import subprocess
        import sys

        # MiniCPM-V-2_6's trust_remote_code model class predates several
        # transformers refactors (tied-weights handling, dtype kwarg rename)
        # and crashes on whatever latest version VLM_Runner.py installed for
        # this session. Its own model card pins this exact version. Scoped
        # to this model only — LLaVA/Qwen2.5-VL never call this method.
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q",
             "transformers==4.40.0", "tokenizers==0.19.1", "huggingface_hub==0.23.0"],
            check=True,
        )
        # transformers is already cached in sys.modules by this point (e.g.
        # bert_score, imported earlier by src/utils/metrics.py, pulls it in)
        # so the fresh install on disk won't take effect until we drop the
        # cached modules and let the next import re-read them from disk.
        for mod_name in list(sys.modules):
            if mod_name.split(".")[0] in ("transformers", "tokenizers", "huggingface_hub"):
                del sys.modules[mod_name]

        import types

        from transformers import AutoModel, AutoTokenizer

        # modeling_navit_siglip.py (part of MiniCPM-V-2_6's remote code) has
        # a module-level `import flash_attn`, used only if flash-attn is
        # actually available at runtime (checked separately, via installed
        # packages, not sys.modules). transformers' trust_remote_code loader
        # does its own naive text scan for import statements and hard-fails
        # if the package can't be imported, even though it's optional here.
        # An earlier fix silenced this by stubbing out check_imports()
        # entirely, but that function also returns the relative imports
        # (e.g. modeling_navit_siglip.py itself) that the loader needs to
        # recursively download — stubbing it broke that download instead.
        # Registering a harmless fake flash_attn module satisfies the
        # import-scan without touching check_imports' relative-import
        # resolution. flash-attn is slow/fragile to build on Colab and
        # isn't needed to run this model.
        if "flash_attn" not in sys.modules:
            sys.modules["flash_attn"] = types.ModuleType("flash_attn")

        hf_id = self.config['model']['hf_model_id']
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            hf_id, trust_remote_code=True, torch_dtype=torch.float16
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

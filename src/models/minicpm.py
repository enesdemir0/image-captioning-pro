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

        import transformers.dynamic_module_utils as dynamic_module_utils
        from transformers import AutoModel, AutoTokenizer

        # modeling_navit_siglip.py (part of MiniCPM-V-2_6's remote code)
        # calls is_flash_attn_2_available() at import time to decide
        # whether to use flash attention — harmless and correctly skipped
        # if the package is missing. But transformers' trust_remote_code
        # loader first runs its own naive text scan (check_imports) that
        # treats any package name mentioned anywhere in the file as a hard
        # requirement, and refuses to even download the file if it's
        # missing. flash-attn is slow/fragile to build on Colab and isn't
        # needed to run this model.
        #
        # Fix that *only* for the duration of check_imports' scan: make it
        # think flash_attn is importable just long enough to pass, then
        # remove the stub immediately. It must not still be in sys.modules
        # once the model's own code actually runs is_flash_attn_2_available()
        # — that function uses importlib.util.find_spec, which raises
        # `flash_attn.__spec__ is None` on a bare stub module instead of
        # cleanly returning "not installed" (this broke an earlier version
        # of this fix that left the stub in place permanently).
        _original_check_imports = dynamic_module_utils.check_imports

        def _check_imports_allow_missing_flash_attn(filename):
            try:
                return _original_check_imports(filename)
            except ImportError as e:
                if "flash_attn" not in str(e):
                    raise
                sys.modules["flash_attn"] = types.ModuleType("flash_attn")
                try:
                    return _original_check_imports(filename)
                finally:
                    del sys.modules["flash_attn"]

        dynamic_module_utils.check_imports = _check_imports_allow_missing_flash_attn

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

import torch
from PIL import Image
from src.models.base_vlm import BaseVLM


class MiniCPMModel(BaseVLM):
    """MiniCPM-V-8B — openbmb/MiniCPM-V-2_6"""

    def load(self):
        import importlib.metadata
        import subprocess
        import sys

        # src/utils/metrics.py imports bert_score at module load time, i.e.
        # before load() ever runs — so bert_score's internal AutoModel/
        # AutoConfig references get bound to whatever transformers version
        # was installed when evaluate.py started. Remember it so unload()
        # can restore it later; see unload() for why that matters.
        self._original_pkg_versions = {
            pkg: importlib.metadata.version(pkg)
            for pkg in ("transformers", "tokenizers", "huggingface_hub")
        }

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
        import transformers.utils as transformers_utils
        import transformers.utils.import_utils as import_utils
        from transformers import AutoModel, AutoTokenizer

        # modeling_navit_siglip.py (part of MiniCPM-V-2_6's remote code)
        # calls is_flash_attn_2_available() at import time to decide
        # whether to use flash attention. flash-attn is slow/fragile to
        # build on Colab and isn't needed to run this model, so force this
        # to report "not available" directly, rather than trying to get
        # transformers' own detection to reach that conclusion on its own:
        #
        # 1. trust_remote_code's loader (check_imports) does a naive text
        #    scan for import statements and hard-fails if a mentioned
        #    package isn't installed — even one only referenced inside an
        #    `if is_flash_attn_2_available():` block. Stub flash_attn into
        #    sys.modules for just the duration of that scan so it passes.
        # 2. Separately, is_flash_attn_2_available() itself calls
        #    importlib.util.find_spec("flash_attn"), which raises
        #    `flash_attn.__spec__ is None` instead of returning False if
        #    flash_attn is ever found sitting in sys.modules without a
        #    real spec — which happens on a warm HF module cache, where
        #    transformers skips re-running check_imports for files it's
        #    already fetched before (so fix #1 above never even runs, but
        #    something upstream may still have left a stub behind from an
        #    earlier attempt). Overriding the function directly sidesteps
        #    find_spec entirely, so it can't crash either way.
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
        transformers_utils.is_flash_attn_2_available = lambda: False
        import_utils.is_flash_attn_2_available = lambda: False

        hf_id = self.config['model']['hf_model_id']
        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            hf_id, trust_remote_code=True, torch_dtype=torch.float16
        )
        self.model = self.model.to(self.config['model'].get('device', 'cuda'))
        self.model.eval()
        print(f"MiniCPM-V loaded: {hf_id}")

    def unload(self):
        import subprocess
        import sys

        # Undo load()'s pinned downgrade now that all captions are generated.
        # Without this, calculate_metrics() (which runs right after this in
        # evaluate.py) crashes inside bert_score: bert_score.utils bound its
        # AutoModel/AutoConfig to the *original* transformers version at
        # import time, but by then transformers has been downgraded and
        # sys.modules cleared, so bert_score's stale classes end up paired
        # with submodules (e.g. configuration_roberta) that get freshly
        # re-imported from the *downgraded* version's files on disk —
        # producing version-mismatch errors like
        # `RobertaConfig object has no attribute 'sub_configs'`. Reinstalling
        # the exact original versions and clearing the cache again makes
        # everything bert_score touches consistent with what it originally
        # bound.
        pins = [f"{pkg}=={version}" for pkg, version in self._original_pkg_versions.items()]
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pins], check=True)
        for mod_name in list(sys.modules):
            if mod_name.split(".")[0] in ("transformers", "tokenizers", "huggingface_hub"):
                del sys.modules[mod_name]

    def _zero_shot(self, image_path: str) -> str:
        prompt_text = self.config['model'].get('prompt', 'Describe this image in one sentence.')
        image = Image.open(image_path).convert("RGB")
        msgs = [{'role': 'user', 'content': [image, prompt_text]}]
        result = self.model.chat(image=None, msgs=msgs, tokenizer=self.tokenizer)
        return result.strip()

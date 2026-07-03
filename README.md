# Open-Source VLM Image Captioning Evaluation

A modular zero-shot evaluation framework for comparing open-source Vision Language Models (VLMs) on image captioning, using the same MS-COCO test split as the encoder-decoder experiments on `main` branch — enabling direct, apples-to-apples comparison.

This branch reproduces and extends the comparative evaluation methodology from:
> *"Comparative Evaluation of Open-Source VLMs on Image Captioning"*, IDAP 2025 (DOI: 10.1109/IDAP68205.2025.11222357)

---

## Models

| Config `name` | Model | HuggingFace ID |
|---|---|---|
| `llava` | LLaVA-7B | `llava-hf/llava-1.5-7b-hf` |
| `minicpm` | MiniCPM-V-8B | `openbmb/MiniCPM-V-2_6` |
| `qwen2_5_vl` | Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` |

---

## Project Structure

```
image-captioning-pro/
├── configs/
│   └── vlm_config.yaml          ← single file that controls everything
├── src/
│   ├── data/
│   │   └── dataset_loader.py    ← loads fixed test split, slices to num_samples
│   ├── models/
│   │   ├── base_vlm.py          ← abstract base class + strategy dispatch
│   │   ├── llava.py             ← LLaVA-7B wrapper
│   │   ├── minicpm.py           ← MiniCPM-V-8B wrapper
│   │   ├── qwen2_5_vl.py        ← Qwen2.5-VL-7B wrapper
│   │   └── __init__.py          ← get_vlm(config) factory function
│   └── utils/
│       ├── config_loader.py     ← loads vlm_config.yaml
│       └── metrics.py           ← BLEU-1/2/3/4, METEOR, ROUGE-L, BERTScore-F1
├── scripts/
│   ├── VLM_Runner.py            ← one-click Colab orchestrator
│   ├── evaluate.py              ← evaluation loop + MLflow logging
│   ├── generate_test_split.py   ← builds the fixed test split + few-shot example pool
│   └── download_data.py         ← downloads MS-COCO annotations
├── tests/
│   ├── conftest.py
│   └── test_vlm_pipeline.py
├── setup.py
└── requirements.txt
```

---

## Configuration Reference — `configs/vlm_config.yaml`

This is the only file you need to touch between experiments.

```yaml
dataset:
  name: "MS-COCO-2014"
  url_annotations: "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
  image_dir: "data/train2014"
  caption_file: "data/annotations/captions_train2014.json"
  image_prefix: "COCO_train2014_"
  subset_size: 0
  test_split_path: "configs/test_split.json"

model:
  name: "qwen2_5_vl"
  hf_model_id: "Qwen/Qwen2.5-VL-7B-Instruct"
  strategy: "zero_shot"
  prompt: "Describe this image in one sentence."
  device: "cuda"
  max_new_tokens: 50
  load_in_8bit: true

evaluation:
  num_samples: 300
  random_seed: 42

mlflow:
  tracking_uri: "https://dagshub.com/enesdemir0/image-captioning-pro.mlflow"
  repo_owner: "enesdemir0"
  repo_name: "image-captioning-pro"
```

### `dataset` section

| Parameter | What it does | Change it? |
|---|---|---|
| `name` | Label only, not used in code | No |
| `url_annotations` | Where `download_data.py` fetches COCO annotations | No |
| `image_dir` | Path to extracted COCO images on local disk | No |
| `caption_file` | Path to COCO annotation JSON | No |
| `image_prefix` | COCO filename prefix — do not touch | No |
| `subset_size` | How many annotations to load when **generating** the test split. `0` = full dataset. Must match `main` branch | No |
| `test_split_path` | Path to the generated fixed split file | No |

### `model` section

| Parameter | What it does | Change it? |
|---|---|---|
| `name` | Selects the model class. Must be one of: `llava`, `minicpm`, `qwen2_5_vl` | **Yes — to switch model** |
| `hf_model_id` | Exact HuggingFace model ID used for download. Must match `name` | **Yes — together with `name`** |
| `strategy` | Inference strategy. `zero_shot` = image + prompt only. `few_shot` = prepends a few in-context (image, caption) example pairs before the target image | **Yes — to test different strategies** |
| `prompt` | The exact text instruction sent to the model alongside the image | **Yes — to experiment with prompts** |
| `device` | `cuda` for GPU, `cpu` for CPU (very slow) | Rarely |
| `max_new_tokens` | Maximum number of tokens the model can generate per caption | Yes, if captions are cut off |
| `load_in_8bit` | 8-bit quantization. Halves VRAM usage. `true` = ~8GB, `false` = ~14GB (fp16) | Yes, based on VRAM |
| `few_shot_k` | Only used when `strategy: few_shot`. How many in-context examples to prepend (capped at the generated pool size, 5) | Yes, to experiment with fewer/more examples |

### `evaluation` section

| Parameter | What it does | Change it? |
|---|---|---|
| `num_samples` | How many images to evaluate. Always takes the **first N** of the fixed split — so every model with the same N uses identical images | **Yes — set to 10 for quick tests, 300 for full runs** |
| `random_seed` | Seed used when generating the fixed split. Must match `main` branch value | **Never** |

### `mlflow` section

| Parameter | What it does | Change it? |
|---|---|---|
| `tracking_uri` | DagsHub MLflow endpoint | No |
| `repo_owner` | DagsHub username | No |
| `repo_name` | DagsHub repo name | No |

---

## MLflow Experiment Naming

Every experiment gets a self-describing name automatically built from the config:

```
VLM_{name}_{hf_model_id}_{num_samples}samples_{strategy}_{quantization}
```

### Examples

| Config | Experiment Name |
|---|---|
| qwen2_5_vl, 300 samples, zero_shot, 8bit | `VLM_qwen2_5_vl_Qwen-Qwen2.5-VL-7B-Instruct_300samples_zero_shot_8bit` |
| llava, 100 samples, zero_shot, fp16 | `VLM_llava_llava-hf-llava-1.5-7b-hf_100samples_zero_shot_fp16` |
| minicpm, 300 samples, few_shot, 8bit | `VLM_minicpm_openbmb-MiniCPM-V-2_6_300samples_few_shot_8bit` |

These sit alongside the `main` branch encoder-decoder experiments (e.g. `ENC_Xception_DEC_LSTM_L3_U512_S0_E50_TF_scaled_dot_GWO`) in the same DagsHub MLflow UI, so you can compare them directly.

---

## Evaluation Metrics

All 7 metrics are logged to MLflow per run:

| Metric | What it measures |
|---|---|
| `BLEU-1` | Unigram precision (exact word overlap) |
| `BLEU-2` | Bigram precision |
| `BLEU-3` | Trigram precision |
| `BLEU-4` | 4-gram precision — standard captioning benchmark metric |
| `METEOR` | Precision + recall with stemming and synonym matching |
| `ROUGE-L` | Longest common subsequence — measures content coverage |
| `BERTScore-F1` | Semantic similarity using contextual embeddings — captures meaning beyond word overlap |

---

## How to Run (Google Colab)

### Step 1 — Change config locally and push
Edit `configs/vlm_config.yaml` on your local machine (change model, num_samples, prompt, etc.), commit, and push to GitHub. Colab always pulls the latest version from the repo.

### Step 2 — Colab cells

```python
# Cell 1 — verify GPU
!nvidia-smi
```

```python
# Cell 2 — mount Drive (for COCO images if you have the zip)
from google.colab import drive
drive.mount('/content/drive')
```

```python
# Cell 3 — clone the branch
!git clone -b feat/open-vlm-captioning https://github.com/enesdemir0/image-captioning-pro.git
%cd image-captioning-pro
```

```python
# Cell 4 — DagsHub auth
import os
os.environ["DAGSHUB_USER_TOKEN"] = "YOUR_TOKEN_HERE"
```

```python
# Cell 5 — run everything
!python scripts/VLM_Runner.py
```

### What VLM_Runner.py does

1. Installs all dependencies
2. Authenticates with DagsHub/MLflow
3. Checks for COCO images → Drive zip → direct download from MS-COCO (~13GB, ~30 min first time)
4. Downloads COCO annotations (~250MB)
5. Generates the fixed test split (and few-shot example pool) if it does not exist yet
6. Loads the model defined in `vlm_config.yaml` and evaluates on `num_samples` images
7. Logs all 7 metrics + summary JSON to MLflow/DagsHub

### To switch between models
Change `name` and `hf_model_id` in `vlm_config.yaml`, push, then run `!git pull` in Colab (no need to re-clone) and re-run Cell 5.

### To update config without re-cloning
```python
# After re-pushing your config changes:
!git pull
!python scripts/evaluate.py   # skip re-installing deps, run eval directly
```

---

## Fair Comparison — What Is and Isn't Guaranteed

### Fair within this branch
The split generation logic is fully deterministic (`random_state=42` in `train_test_split`, `np.random.seed(42)` for sampling), so every model evaluated against the same underlying COCO images sees the same 300 test samples and the same few-shot example pool. Setting `num_samples: 10` always uses the **first 10** entries, never reshuffled, so this holds regardless of `num_samples`. The same logic applies to the sample images logged to MLflow under `samples/` — `sample_0.png` .. `sample_4.png` point at the same underlying images across every model run.

### Fair against main branch (best-effort, not guaranteed)
The main branch generates its test split dynamically at runtime from the full COCO annotation file. This branch tries to match it by reproducing the same split logic in `scripts/generate_test_split.py`:

- Same annotation file (MS-COCO 2014 train)
- Same `random_state=42`, `test_size=0.15` in `train_test_split`
- Same `np.random.seed(42)` sampling of 300 from the test pool
- Same caption preprocessing: lowercase, punctuation removed, whitespace normalised — mirrors `clean_caption()` in `main:src/data/text_processor.py` exactly

However a true apples-to-apples comparison is not possible because:
- Main branch model was **trained** on COCO training images (in-distribution)
- VLMs here are evaluated **zero-shot** (never specifically trained on COCO)
- These are fundamentally different paradigms — the metric gap reflects that, not just model quality

---

## Adding a New Model

1. Create `src/models/yourmodel.py` implementing `BaseVLM`:

```python
from src.models.base_vlm import BaseVLM

class YourModel(BaseVLM):
    def load(self):
        # load self.model and self.processor

    def _zero_shot(self, image_path: str) -> str:
        # return generated caption string
```

2. Register it in `src/models/__init__.py`:

```python
from src.models.yourmodel import YourModel

_REGISTRY = {
    ...
    'yourmodel': YourModel,
}
```

3. Update `vlm_config.yaml`:

```yaml
model:
  name: "yourmodel"
  hf_model_id: "org/model-name-on-huggingface"
```

---

## Few-Shot Strategy

All three models (`llava`, `minicpm`, `qwen2_5_vl`) implement `_few_shot()`. Set in config:

```yaml
model:
  strategy: "few_shot"
  few_shot_k: 2       # how many in-context examples to prepend
```

### Where the examples come from
`scripts/generate_test_split.py` samples a small pool of examples (`FEW_SHOT_POOL_SIZE = 5`)
from the **train** portion of the split — never from the 300 held-out test images — using a
fixed seed (`random_seed + 1`) so the pool is deterministic too.
`dataset_loader.load_few_shot_examples()` then returns the first `few_shot_k` of that pool.

`evaluate.py` loads these once before the eval loop (only when `strategy: few_shot`) and hands
them to the model as `vlm.few_shot_examples`. Each model's `_few_shot()` builds a multi-turn
prompt: one (image, caption) turn per example, followed by the target image with no caption,
using whatever prompt/chat format that model expects (LLaVA: `USER:`/`ASSISTANT:` turns,
Qwen2.5-VL: chat-template messages, MiniCPM: multi-turn `model.chat()` msgs).

The experiment name automatically reflects the strategy: `..._few_shot_...`

### Adding a New Strategy

Implement `_<name>()` in your model class, add a branch in `BaseVLM.generate_caption()`'s
dispatch, and set `strategy: "<name>"` in config.

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
│   ├── vlm_config.yaml          ← single file that controls everything
│   └── test_split.json          ← 300-sample test split, regenerated each Colab
│                                   session by generate_test_split.py (NOT yet
│                                   committed to git — see note below)
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
│   ├── generate_test_split.py   ← run once to create configs/test_split.json
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
| `test_split_path` | Path to the committed fixed split file | No |

### `model` section

| Parameter | What it does | Change it? |
|---|---|---|
| `name` | Selects the model class. Must be one of: `llava`, `minicpm`, `qwen2_5_vl` | **Yes — to switch model** |
| `hf_model_id` | Exact HuggingFace model ID used for download. Must match `name` | **Yes — together with `name`** |
| `strategy` | Inference strategy. `zero_shot` = image + prompt only. `few_shot` = coming later | **Yes — to test different strategies** |
| `prompt` | The exact text instruction sent to the model alongside the image | **Yes — to experiment with prompts** |
| `device` | `cuda` for GPU, `cpu` for CPU (very slow) | Rarely |
| `max_new_tokens` | Maximum number of tokens the model can generate per caption | Yes, if captions are cut off |
| `load_in_8bit` | 8-bit quantization. Halves VRAM usage. `true` = ~8GB, `false` = ~14GB (fp16) | Yes, based on VRAM |

### `evaluation` section

| Parameter | What it does | Change it? |
|---|---|---|
| `num_samples` | How many images to evaluate. Always takes the **first N** from `test_split.json` — so every model with the same N uses identical images | **Yes — set to 10 for quick tests, 300 for full runs** |
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
5. Generates `configs/test_split.json` if it does not exist (run once, then committed to git)
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

### Fair within this branch (currently true in practice, not yet enforced by git)
`configs/test_split.json` is **not committed to git yet** — `VLM_Runner.py` regenerates it fresh each Colab session via `scripts/generate_test_split.py` whenever the file isn't already present locally. That script is fully deterministic (`random_state=42` in `train_test_split`, `np.random.seed(42)` for sampling), so it reproduces the exact same 300 samples every time **as long as the same COCO image files are on disk at generation time**.

In practice, all evaluations so far (LLaVA, Qwen2.5-VL, MiniCPM) sourced images from the same unchanging Google Drive zip, so regeneration has produced identical splits each session — all three models were evaluated on the same 300 images with the same reference captions. Setting `num_samples: 10` always uses the **first 10** entries of that split, never reshuffled, so this holds regardless of `num_samples`.

This also means the sample images logged to MLflow under `samples/` (the first 5 `results/samples/sample_N.png` files, each showing REAL vs PRED caption) are the same 5 images across every model run — `evaluate.py` saves `sample_0.png` .. `sample_4.png` from `samples[0:5]`, and since every session regenerates the identical split, those indices always point at the same underlying images regardless of which model produced the prediction.

> **To do:** this guarantee currently depends on always sourcing images from the same Drive zip — nothing in the code enforces it. Commit the generated `configs/test_split.json` to git (see `scripts/generate_test_split.py`'s own printed reminder) so the split is pinned by file content rather than by environment discipline.

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

## Adding a New Strategy (e.g. Few-Shot)

Implement `_few_shot()` in your model class:

```python
def _few_shot(self, image_path: str) -> str:
    # build prompt with example image-caption pairs
    # return generated caption string
```

Then set in config:

```yaml
model:
  strategy: "few_shot"
```

The experiment name will automatically reflect this: `..._few_shot_...`

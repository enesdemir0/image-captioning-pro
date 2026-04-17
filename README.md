# Image Captioning Engine — MS-COCO

A production-ready, fully modular Encoder-Decoder image captioning framework built on TensorFlow, tracked with MLflow/DagsHub, and designed for reproducible research experiments on the MS-COCO 2014 dataset.

---

## Architecture Overview

```
Image (JPEG)
    │
    ▼
┌──────────────────────────────────┐
│  CNN_Encoder  (frozen backbone)  │
│  Xception / InceptionV3 / VGG16  │
│  → Dense bridge → (B, N, units)  │
└──────────────────┬───────────────┘
                   │ spatial feature grid
                   ▼
┌──────────────────────────────────┐
│  Attention Mechanism             │
│  ┌─────────┐  ┌────────────────┐ │
│  │   RAN   │  │ GlobalAttention│ │
│  │ (3×3    │  │ Bahdanau / Dot │ │
│  │  Conv)  │  │ / Scaled-Dot   │ │
│  └─────────┘  └────────────────┘ │
│  → context vector (B, units)     │
└──────────────────┬───────────────┘
                   │
                   ▼
┌──────────────────────────────────┐
│  RNN_Decoder  (stacked)          │
│  Embedding → Concat(ctx, embed)  │
│  → StackedRNN (LSTM/GRU, L deep) │
│  → Dense → logits (vocab_size)   │
└──────────────────────────────────┘
```

**Training features:**
- Teacher Forcing (toggle via `use_teacher_forcing` in `config.yaml`)
- Grey Wolf Optimizer (GWO) learning rate schedule (toggle via `optimizer_type: greywolf`)
- Dual checkpoint persistence: Colab session cache (`./checkpoints/`) + Google Drive
- Frozen tokenizer JSON saved on every checkpoint — word indices never shift between sessions

---

## The DNA Naming Convention

Every model, experiment, and checkpoint file is named with a single deterministic string:

```
ENC_{enc}_DEC_{dec}_L{layers}_U{units}_S{subset}_E{epochs}_{tf_val}_{attn}_{opt}
```

| Token | Meaning | Example |
|---|---|---|
| `ENC_` | CNN backbone | `ENC_Xception` |
| `DEC_` | RNN cell type | `DEC_LSTM` |
| `L` | Number of RNN layers | `L3` |
| `U` | Hidden units per layer | `U512` |
| `S` | Annotation subset size | `S500` |
| `E` | Total training epochs | `E10` |
| `TF`/`Base` | Teacher Forcing on/off | `TF` |
| `{attn}` | Attention type | `dot` / `ran` / `bahdanau` |
| `{opt}` | Optimizer | `Adam` / `GWO` |

**Example:**
```
ENC_Xception_DEC_LSTM_L3_U512_S500_E10_TF_dot_Adam
```

This string is used as the MLflow experiment name, checkpoint file prefix, tokenizer JSON filename, and evaluation summary filename — ensuring every artifact is traceable back to its exact hyperparameter configuration.

---

## Project Structure

```
image-captioning-pro/
├── configs/
│   └── config.yaml              ← All hyperparameters live here
├── src/
│   ├── data/
│   │   ├── text_processor.py    ← Caption cleaning, tokenisation, JSON persistence
│   │   ├── dataset_loader.py    ← MS-COCO annotation loader + tf.data pipeline
│   │   └── image_processor.py  ← Model-specific image normalisation
│   ├── models/
│   │   ├── encoder.py           ← CNN_Encoder (transfer learning)
│   │   ├── decoder.py           ← RNN_Decoder (multi-layer, pluggable attention)
│   │   └── attention.py         ← RegionAttention (RAN) + GlobalAttention
│   ├── training/
│   │   ├── trainer.py           ← CaptionTrainer (train/val steps, GWO)
│   │   └── train.py             ← Main training script
│   └── utils/
│       ├── config_loader.py     ← YAML config reader
│       └── metrics.py           ← BLEU, METEOR, ROUGE-L
├── scripts/
│   ├── evaluate.py              ← Evaluation + heatmap generation
│   ├── download_data.py         ← MS-COCO annotation downloader
│   └── Image_Captioning_Runner.py ← One-click Colab orchestrator
├── tests/                       ← Pytest suite
└── requirements.txt
```

---

## Google Colab Setup

### Step 1 — Mount Drive and clone
```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/enesdemir0/image-captioning-pro.git
%cd image-captioning-pro
!pip install -e . -q
```

### Step 2 — Set credentials
```python
import os
os.environ["DAGSHUB_USER_TOKEN"] = "YOUR_DAGSHUB_TOKEN"
```

### Step 3 — Download MS-COCO annotations
```python
!python scripts/download_data.py
```
Place training images in `data/train2014/` (or adjust `image_dir` in `config.yaml`).

### Step 4 — Configure your experiment
Edit `configs/config.yaml` to select your backbone, cell type, attention mechanism, and training budget.

### Step 5 — Train
```python
!python -m src.training.train
```

### Step 6 — Evaluate
```python
!python scripts/evaluate.py
```
Weights are automatically located in `./checkpoints/` (Colab session) first, then Google Drive. The tokenizer is loaded from the frozen JSON — no re-fitting required.

---

## Key Features

### Teacher Forcing
Controlled by `use_teacher_forcing: True/False` in `config.yaml`.

When enabled, the ground-truth token at time step *t* is fed as input at step *t+1* during training, stabilising gradients and accelerating convergence. During validation the decoder always uses its own predictions (autoregressive mode), providing an honest performance signal.

The model ID encodes this choice as `TF` (on) or `Base` (off), ensuring that Teacher-Forcing and free-running experiments are tracked as separate MLflow experiments.

### Grey Wolf Optimizer (GWO)
Set `optimizer_type: greywolf` in `config.yaml`.

Inspired by the GWO metaheuristic (Mirjalili et al. 2014), the learning rate follows a linear decay from `learning_rate` to `1e-6` over the full training run — mirroring the GWO control parameter *a* which decreases from 2 to 0. This schedule produces smooth convergence without requiring manual LR tuning.

```python
new_lr = initial_lr × (1 − epoch / total_epochs)
```

The GWO schedule is applied once per epoch, before the training step loop.

### Region Attention Network (RAN)
Set `attention_type: ran` in `config.yaml`.

Before computing attention scores, RAN applies a 3×3 Conv2D layer over the spatial feature grid (e.g. 10×10 for Xception). This allows the model to score *regions* of the image rather than isolated point features, improving localisation of relevant objects for each generated word.

Feature flow:
```
(batch, 100, 512)
  → reshape to (batch, 10, 10, 512)
  → Conv2D(3×3, 512, same)
  → reshape to (batch, 100, 512)
  → additive attention scoring
```

---

## Interpretability — Attention Heatmaps

`scripts/evaluate.py` automatically generates overlaid attention heatmaps for the first 5 test images and logs them as MLflow artifacts.

**Heatmap pipeline:**
1. **Temporal averaging** — per-step attention weight maps are averaged across all generated tokens to produce a single spatial importance map.
2. **Bicubic upsampling** — `cv2.resize(..., interpolation=cv2.INTER_CUBIC)` scales the (grid × grid) map to the full image resolution with smooth, artefact-free interpolation.
3. **Gaussian smoothing** — `cv2.GaussianBlur(heatmap, (21, 21), 0)` blurs sharp activation boundaries, producing a natural 'glow' around attended regions.
4. **Min-Max normalisation** — the map is linearly rescaled to [0, 1], ensuring the full dynamic range of the jet colormap is always used regardless of raw attention magnitude.
5. **Jet colormap overlay** — the normalised map is coloured with matplotlib's `jet` palette (blue → green → red) and alpha-blended at 50/50 with the original image.

The result shows which spatial regions of the image the decoder was attending to when generating each word in the caption. High-activation (red/yellow) areas correspond to the model's focal regions.

---

## Experiment Tracking

All experiments are tracked on [DagsHub](https://dagshub.com/enesdemir0/image-captioning-pro) via MLflow.

Logged per run:
- **Parameters:** all `model` and `training` config entries
- **Metrics (training):** `train_loss`, `val_loss` per epoch
- **Metrics (evaluation):** `BLEU-1`, `BLEU-2`, `BLEU-3`
- **Artifacts:** attention heatmap PNGs, evaluation summary JSON

---

## Configuration Reference

```yaml
dataset:
  vocab_size: 10000           # Top-N most frequent tokens
  max_caption_length: 25      # Sequence length (pad/truncate)
  subset_size: 500            # 0 = full dataset

model:
  encoder_name: "Xception"    # Xception | InceptionV3 | VGG16 | ResNet50
  decoder_type: "LSTM"        # LSTM | GRU
  num_layers: 3               # RNN stack depth
  units: 512                  # Hidden units (encoder bridge + decoder)
  embedding_dim: 256          # Token embedding size
  attention_type: "dot"       # bahdanau | dot | scaled_dot | ran | None

training:
  epochs: 10
  batch_size: 256
  learning_rate: 0.0005
  use_teacher_forcing: True
  optimizer_type: "adam"      # adam | greywolf
  checkpoint_path: "/content/drive/MyDrive/Image_Captioning_Models/checkpoints/"
```

---

## Citation / Acknowledgements

- MS-COCO: Lin et al. (2014) — *Microsoft COCO: Common Objects in Context*
- Bahdanau Attention: Bahdanau et al. (2015) — *Neural Machine Translation by Jointly Learning to Align and Translate*
- GWO: Mirjalili et al. (2014) — *Grey Wolf Optimizer*
- Teacher Forcing: Williams & Zipser (1989)

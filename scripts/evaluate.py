"""
evaluate.py — Standalone evaluation script for the Image Captioning Engine.

Session-Aware Loading:
  1. Checks ./checkpoints/ (Colab session cache) for instant post-training access.
  2. Falls back to Google Drive if local weights are absent.

Tokenizer is always loaded from the frozen JSON — never re-fitted.
This guarantees identical word↔index mapping between training and evaluation.
"""

import os
import json
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mlflow
import dagshub
from mlflow.tracking import MlflowClient
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import nltk

from src.utils.config_loader import load_config
from src.data.dataset_loader import DataLoader
from src.data.image_processor import ImageProcessor
from src.models.encoder import CNN_Encoder
from src.models.decoder import RNN_Decoder

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False
    print("Warning: opencv-python not found. Install it for professional heatmap quality.")

# Encoder output grid sizes  (H × W spatial patches per backbone)
_NUM_FEATURES = {"Xception": 100, "InceptionV3": 64, "VGG16": 49, "ResNet50": 49}


# ─── Checkpoint Resolution ────────────────────────────────────────────────────

def _resolve_paths(config, model_id):
    """
    Returns (enc_path, dec_path, tokenizer_path).
    Checks local Colab ./checkpoints/ first, then Google Drive.
    """
    local_dir = "./checkpoints/"
    drive_dir = config['training']['checkpoint_path']

    enc_local = os.path.join(local_dir, f"{model_id}_enc.weights.h5")
    dec_local = os.path.join(local_dir, f"{model_id}_dec.weights.h5")
    tok_local = os.path.join(local_dir, f"{model_id}_tokenizer.json")

    enc_drive = os.path.join(drive_dir, f"{model_id}_enc.weights.h5")
    dec_drive = os.path.join(drive_dir, f"{model_id}_dec.weights.h5")
    tok_drive = os.path.join(drive_dir, f"{model_id}_tokenizer.json")

    if os.path.exists(enc_local) and os.path.exists(dec_local):
        print(f"Session cache hit — loading weights from {local_dir}")
        enc_path, dec_path = enc_local, dec_local
    elif os.path.exists(enc_drive) and os.path.exists(dec_drive):
        print(f"Loading weights from Google Drive: {drive_dir}")
        enc_path, dec_path = enc_drive, dec_drive
    else:
        raise FileNotFoundError(
            f"No checkpoint found for model '{model_id}'.\n"
            f"  Checked local : {local_dir}\n"
            f"  Checked Drive : {drive_dir}\n"
            f"Run training first."
        )

    if os.path.exists(tok_local):
        tok_path = tok_local
    elif os.path.exists(tok_drive):
        tok_path = tok_drive
    else:
        raise FileNotFoundError(
            f"Tokenizer JSON not found for '{model_id}'.\n"
            f"  Checked local : {tok_local}\n"
            f"  Checked Drive : {tok_drive}"
        )

    return enc_path, dec_path, tok_path


# ─── Caption Generation ───────────────────────────────────────────────────────

def generate_caption(image_path, encoder, decoder, text_processor, image_processor, config):
    """
    Greedy inference pass.
    Returns:
        words            : list[str]  — generated caption tokens
        attn_weights_list: list[ndarray] — per-step attention maps
        orig_image       : ndarray   — original uint8 image (H, W, 3)
    """
    # Original image for heatmap display (read separately, before normalization)
    raw = tf.io.read_file(image_path)
    orig_image = tf.image.decode_jpeg(raw, channels=3).numpy().astype(np.uint8)

    # Preprocessed tensor for model inference
    img_tensor = image_processor.preprocess_image(image_path)[0]
    img_tensor = tf.expand_dims(img_tensor, 0)

    features = encoder(img_tensor)
    hidden   = decoder.init_decoder_state(features)

    start_idx = text_processor.tokenizer.word_index.get('<start>', 1)
    end_idx   = text_processor.tokenizer.word_index.get('<end>', 2)
    dec_input = tf.expand_dims([start_idx], 0)

    words             = []
    attn_weights_list = []

    for _ in range(config['dataset']['max_caption_length']):
        predictions, hidden, attn_weights = decoder(dec_input, features, hidden)

        if attn_weights is not None:
            attn_weights_list.append(attn_weights.numpy().squeeze())

        predicted_id = int(tf.argmax(predictions[0]).numpy())
        word = text_processor.tokenizer.index_word.get(predicted_id, '<unk>')

        if word == '<end>':
            break

        words.append(word)
        dec_input = tf.expand_dims([predicted_id], 0)

    return words, attn_weights_list, orig_image


# ─── Attention Heatmap ────────────────────────────────────────────────────────

def generate_attention_heatmap(orig_image, attn_weights_list, caption_words,
                               output_path):
    """
    Professional attention heatmap overlay.

    Pipeline:
      1. Average all per-step attention maps → single (num_features,) map.
      2. Reshape to square grid (e.g. 10×10 for Xception).
      3. Bicubic upsampling to image resolution (cv2.INTER_CUBIC).
      4. Gaussian smoothing (21×21 kernel) for organic 'glow' effect.
      5. Min-Max normalisation → [0, 1].
      6. Jet colormap → alpha-blend at 50/50 with original image.
    """
    if not attn_weights_list:
        print(f"  No attention weights — heatmap skipped for {output_path}")
        return

    # Step 1 — average across time steps
    attn_map = np.mean(np.stack(attn_weights_list, axis=0), axis=0).flatten()

    # Step 2 — reshape to square grid
    num_features = len(attn_map)
    grid_size    = int(np.sqrt(num_features))
    attn_map     = attn_map[:grid_size * grid_size].reshape(grid_size, grid_size)

    h, w = orig_image.shape[:2]
    orig_float = orig_image.astype(np.float32) / 255.0

    # Step 3 & 4 — upsampling + Gaussian blur
    if _CV2:
        heatmap = cv2.resize(
            attn_map.astype(np.float32), (w, h),
            interpolation=cv2.INTER_CUBIC
        )
        heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
    else:
        # Fallback: TF bilinear resize
        heatmap = tf.image.resize(
            attn_map[:, :, np.newaxis], (h, w)
        ).numpy().squeeze().astype(np.float32)

    # Step 5 — Min-Max normalisation
    lo, hi = heatmap.min(), heatmap.max()
    if hi - lo > 1e-8:
        heatmap = (heatmap - lo) / (hi - lo)

    # Step 6 — Jet colormap + blend
    colored = plt.get_cmap('jet')(heatmap)[:, :, :3]   # drop alpha
    overlay = np.clip(0.5 * orig_float + 0.5 * colored, 0.0, 1.0)

    # Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].imshow(orig_image)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis('off')

    axes[1].imshow(overlay)
    axes[1].set_title(f"Region Attention Map\n\"{' '.join(caption_words)}\"", fontsize=10)
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Heatmap saved → {output_path}")


# ─── BLEU Metrics ─────────────────────────────────────────────────────────────

def compute_bleu_scores(references, hypotheses):
    """
    Computes corpus BLEU-1, BLEU-2, BLEU-3.
    references  : list[str] — ground-truth captions (plain text)
    hypotheses  : list[str] — predicted captions (plain text)
    """
    smoother      = SmoothingFunction().method1
    refs_tok      = [[r.split()] for r in references]
    hyps_tok      = [h.split()   for h in hypotheses]

    b1 = corpus_bleu(refs_tok, hyps_tok, weights=(1, 0, 0, 0),             smoothing_function=smoother)
    b2 = corpus_bleu(refs_tok, hyps_tok, weights=(0.5, 0.5, 0, 0),         smoothing_function=smoother)
    b3 = corpus_bleu(refs_tok, hyps_tok, weights=(1/3, 1/3, 1/3, 0),       smoothing_function=smoother)

    return b1, b2, b3


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    config = load_config()

    # ── DNA Model ID (IMMUTABLE NAMING CONVENTION) ────────────────────────────
    enc    = config['model']['encoder_name']
    dec    = config['model']['decoder_type']
    layers = config['model']['num_layers']
    units  = config['model']['units']
    subset = config['dataset']['subset_size']
    epochs = config['training']['epochs']
    attn   = config['model'].get('attention_type', 'None')
    opt    = "Adam" if config['training'].get('optimizer_type') == "adam" else "GWO"
    tf_val = "TF"   if config['training']['use_teacher_forcing'] else "Base"

    model_id = f"ENC_{enc}_DEC_{dec}_L{layers}_U{units}_S{subset}_E{epochs}_{tf_val}_{attn}_{opt}"
    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_id}")
    print(f"{'='*60}\n")

    # ── MLflow / DagsHub Setup ────────────────────────────────────────────────
    dagshub.init(
        repo_owner=config['mlflow']['repo_owner'],
        repo_name=config['mlflow']['repo_name'],
        mlflow=True
    )
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    client = MlflowClient()
    exp = client.get_experiment_by_name(model_id)
    if exp and exp.lifecycle_stage == 'deleted':
        client.restore_experiment(exp.experiment_id)
    mlflow.set_experiment(model_id)

    # ── Session-Aware Checkpoint Resolution ──────────────────────────────────
    enc_path, dec_path, tokenizer_path = _resolve_paths(config, model_id)

    # ── Data Pipeline (test split only) ──────────────────────────────────────
    loader = DataLoader(config)
    img_paths, captions = loader.load_annotations()
    _, _, (test_imgs, test_caps) = loader.split_data(img_paths, captions)

    # Load frozen tokenizer — NO re-fitting
    loader.text_processor.load_tokenizer(tokenizer_path)
    loader._tokenizer_fitted = True
    print(f"Vocabulary size: {len(loader.text_processor.tokenizer.word_index)} tokens")
    print(f"Test set size  : {len(test_imgs)} images\n")

    # ── Build & Load Models ───────────────────────────────────────────────────
    encoder       = CNN_Encoder(config)
    decoder       = RNN_Decoder(config)
    image_proc    = ImageProcessor(config)
    num_f         = _NUM_FEATURES.get(enc, 100)

    encoder(tf.zeros((1, 299, 299, 3)))
    decoder(
        tf.zeros((1, 1)),
        tf.zeros((1, num_f, units)),
        decoder.init_decoder_state(tf.zeros((1, num_f, units)))
    )
    encoder.load_weights(enc_path)
    decoder.load_weights(dec_path)
    print(f"Weights loaded successfully from checkpoint.\n")

    # ── Inference Loop ────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)

    references  = []
    hypotheses  = []
    heatmap_idx = 0

    with mlflow.start_run(run_name="Evaluation_Session"):
        mlflow.log_param("model_id",  model_id)
        mlflow.log_param("test_size", len(test_imgs))

        for i, (img_path, ref_cap) in enumerate(zip(test_imgs, test_caps)):
            words, attn_weights, orig_image = generate_caption(
                img_path, encoder, decoder,
                loader.text_processor, image_proc, config
            )

            ref_clean = ref_cap.replace('<start>', '').replace('<end>', '').strip()
            pred_str  = ' '.join(words)

            references.append(ref_clean)
            hypotheses.append(pred_str)

            # Heatmaps for the first 5 images that have attention weights
            if heatmap_idx < 5 and attn_weights:
                heatmap_path = f"results/heatmap_{heatmap_idx:02d}_{i:04d}.png"
                generate_attention_heatmap(orig_image, attn_weights, words, heatmap_path)
                mlflow.log_artifact(heatmap_path)
                heatmap_idx += 1

            if i % 50 == 0:
                print(f"  [{i:>4}/{len(test_imgs)}] {pred_str}")

        # ── BLEU Scores ───────────────────────────────────────────────────────
        b1, b2, b3 = compute_bleu_scores(references, hypotheses)

        print(f"\n{'='*60}")
        print(f"  EVALUATION RESULTS  —  {model_id}")
        print(f"{'='*60}")
        print(f"  BLEU-1 : {b1:.4f}")
        print(f"  BLEU-2 : {b2:.4f}")
        print(f"  BLEU-3 : {b3:.4f}")
        print(f"{'='*60}\n")

        mlflow.log_metrics({"BLEU-1": b1, "BLEU-2": b2, "BLEU-3": b3})

        # Persist summary
        summary_path = f"results/{model_id}_eval_summary.json"
        summary = {
            "model_id":  model_id,
            "test_size": len(test_imgs),
            "BLEU-1":    round(b1, 4),
            "BLEU-2":    round(b2, 4),
            "BLEU-3":    round(b3, 4),
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(summary_path)
        print(f"Summary saved → {summary_path}")


if __name__ == "__main__":
    main()

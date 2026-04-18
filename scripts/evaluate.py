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
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk

from src.utils.config_loader import load_config
from src.data.dataset_loader import DataLoader
from src.data.image_processor import ImageProcessor
from src.models.encoder import CNN_Encoder
from src.models.decoder import RNN_Decoder

# --- NLTK Resources ---
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

try:
    import cv2
    _CV2 = True
except ImportError:
    _CV2 = False

_NUM_FEATURES = {"Xception": 100, "InceptionV3": 64, "VGG16": 49, "ResNet50": 49}

# ─── Checkpoint Resolution ────────────────────────────────────────────────────

def _resolve_paths(config, model_id):
    local_dir = "./checkpoints/"
    drive_dir = config['training']['checkpoint_path']
    enc_local = os.path.join(local_dir, f"{model_id}_enc.weights.h5")
    dec_local = os.path.join(local_dir, f"{model_id}_dec.weights.h5")
    tok_local = os.path.join(local_dir, f"{model_id}_tokenizer.json")
    enc_drive = os.path.join(drive_dir, f"{model_id}_enc.weights.h5")
    dec_drive = os.path.join(drive_dir, f"{model_id}_dec.weights.h5")
    tok_drive = os.path.join(drive_dir, f"{model_id}_tokenizer.json")

    if os.path.exists(enc_local) and os.path.exists(dec_local):
        print(f"✅ Session cache hit — loading from {local_dir}")
        enc_path, dec_path = enc_local, dec_local
    elif os.path.exists(enc_drive) and os.path.exists(dec_drive):
        print(f"✅ Loading from Google Drive: {drive_dir}")
        enc_path, dec_path = enc_drive, dec_drive
    else:
        raise FileNotFoundError(f"No weights found for {model_id}")

    tok_path = tok_local if os.path.exists(tok_local) else tok_drive
    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"Tokenizer JSON missing for {model_id}")

    return enc_path, dec_path, tok_path

# ─── Caption Generation ───────────────────────────────────────────────────────

def generate_caption(image_path, encoder, decoder, text_processor, image_processor, config):
    raw = tf.io.read_file(image_path)
    orig_image = tf.image.decode_jpeg(raw, channels=3).numpy().astype(np.uint8)
    img_tensor = image_processor.preprocess_image(image_path)[0]
    img_tensor = tf.expand_dims(img_tensor, 0)

    features = encoder(img_tensor)
    hidden   = decoder.init_decoder_state(features)
    start_idx = text_processor.tokenizer.word_index.get('<start>', 1)
    dec_input = tf.expand_dims([start_idx], 0)

    words, attn_weights_list = [], []
    for _ in range(config['dataset']['max_caption_length']):
        predictions, hidden, attn_weights = decoder(dec_input, features, hidden)
        if attn_weights is not None:
            attn_weights_list.append(attn_weights.numpy().squeeze())
        predicted_id = int(tf.argmax(predictions[0]).numpy())
        word = text_processor.tokenizer.index_word.get(predicted_id, '<unk>')
        if word == '<end>': break
        words.append(word)
        dec_input = tf.expand_dims([predicted_id], 0)
    return words, attn_weights_list, orig_image

# ─── Attention Heatmap (Updated to show Real Caption) ─────────────────────────

def generate_attention_heatmap(orig_image, attn_weights_list, caption_words, real_caption, output_path):
    if not attn_weights_list: return
    attn_map = np.mean(np.stack(attn_weights_list, axis=0), axis=0).flatten()
    grid_size = int(np.sqrt(len(attn_map)))
    attn_map = attn_map[:grid_size * grid_size].reshape(grid_size, grid_size)
    h, w = orig_image.shape[:2]
    
    if _CV2:
        heatmap = cv2.resize(attn_map.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
    else:
        heatmap = tf.image.resize(attn_map[:, :, np.newaxis], (h, w)).numpy().squeeze()

    if heatmap.max() - heatmap.min() > 1e-8:
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    colored = plt.get_cmap('jet')(heatmap)[:, :, :3]
    overlay = np.clip(0.5 * (orig_image.astype(np.float32)/255.0) + 0.5 * colored, 0.0, 1.0)

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    # Subplot 0: Original + Real
    axes[0].imshow(orig_image)
    axes[0].set_title(f"Original Image\nREAL: {real_caption}", fontsize=10, pad=10, fontweight='bold')
    axes[0].axis('off')

    # Subplot 1: Heatmap + Prediction
    axes[1].imshow(overlay)
    axes[1].set_title(f"Attention Map\nPRED: {' '.join(caption_words)}", fontsize=10, pad=10, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight'); plt.close(fig)
    print(f"  Heatmap saved → {output_path}")

# ─── Advanced Metrics ─────────────────────────────────────────────────────────

def calculate_corpus_metrics(references, hypotheses):
    """Calculates BLEU 1-4, METEOR, and ROUGE-L for the whole set."""
    smoother = SmoothingFunction().method1
    refs_tok = [[r.split()] for r in references]
    hyps_tok = [h.split() for h in hypotheses]

    # BLEU Scores
    b1 = corpus_bleu(refs_tok, hyps_tok, weights=(1, 0, 0, 0), smoothing_function=smoother)
    b2 = corpus_bleu(refs_tok, hyps_tok, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother)
    b3 = corpus_bleu(refs_tok, hyps_tok, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoother)
    b4 = corpus_bleu(refs_tok, hyps_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)

    # METEOR and ROUGE-L (Averaged over sentences)
    meteor_scores, rouge_scores = [], []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    for r, h in zip(references, hypotheses):
        meteor_scores.append(meteor_score([r.split()], h.split()))
        rouge_scores.append(scorer.score(r, h)['rougeL'].fmeasure)

    return b1, b2, b3, b4, np.mean(meteor_scores), np.mean(rouge_scores)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    config = load_config()
    enc, dec = config['model']['encoder_name'], config['model']['decoder_type']
    layers, units = config['model']['num_layers'], config['model']['units']
    subset, epochs = config['dataset']['subset_size'], config['training']['epochs']
    attn = config['model'].get('attention_type', 'None')
    opt = "Adam" if config['training'].get('optimizer_type') == "adam" else "GWO"
    tf_val = "TF" if config['training']['use_teacher_forcing'] else "Base"

    model_id = f"ENC_{enc}_DEC_{dec}_L{layers}_U{units}_S{subset}_E{epochs}_{tf_val}_{attn}_{opt}"
    
    dagshub.init(repo_owner="enesdemir0", repo_name="image-captioning-pro", mlflow=True)
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(model_id)

    enc_path, dec_path, tokenizer_path = _resolve_paths(config, model_id)

    loader = DataLoader(config)
    img_paths, captions = loader.load_annotations()
    (tr_i, tr_c), _, (test_imgs, test_caps) = loader.split_data(img_paths, captions)
    
    # --- SPEED FIX: Scientific Random Sampling (300 samples) ---
    total_available = len(test_imgs)
    eval_limit = min(300, total_available)
    
    indices = np.arange(total_available)
    np.random.seed(42)
    np.random.shuffle(indices)
    selected_indices = indices[:eval_limit]
    
    test_imgs_sampled = [test_imgs[i] for i in selected_indices]
    test_caps_sampled = [test_caps[i] for i in selected_indices]
    
    print(f"🧪 Scientific Sampling: Evaluating on {eval_limit} random unseen samples.")

    loader.text_processor.load_tokenizer(tokenizer_path)

    encoder, decoder, image_proc = CNN_Encoder(config), RNN_Decoder(config), ImageProcessor(config)
    encoder(tf.zeros((1, 299, 299, 3)))
    num_f = _NUM_FEATURES.get(enc, 100)
    decoder(tf.zeros((1, 1)), tf.zeros((1, num_f, units)), decoder.init_decoder_state(tf.zeros((1, num_f, units))))
    encoder.load_weights(enc_path); decoder.load_weights(dec_path)

    os.makedirs("results", exist_ok=True)
    references, hypotheses = [], []
    heatmap_idx = 0

    with mlflow.start_run(run_name="Scientific_Test_Evaluation"):
        for i, (img_path, ref_cap) in enumerate(zip(test_imgs_sampled, test_caps_sampled)):
            words, attn_weights, orig_image = generate_caption(img_path, encoder, decoder, loader.text_processor, image_proc, config)
            ref_clean = ref_cap.replace('<start>', '').replace('<end>', '').strip()
            pred_str = ' '.join(words)
            references.append(ref_clean)
            hypotheses.append(pred_str)

            if heatmap_idx < 5 and attn_weights:
                heatmap_path = f"results/heatmap_{i}.png"
                # Pass ref_clean to show the REAL caption on the plot
                generate_attention_heatmap(orig_image, attn_weights, words, ref_clean, heatmap_path)
                mlflow.log_artifact(heatmap_path)
                heatmap_idx += 1
            
            if i % 50 == 0: 
                print(f"  Processed {i}/{eval_limit}")

        # --- CALCULATE ALL 6 METRICS ---
        b1, b2, b3, b4, m, r = calculate_corpus_metrics(references, hypotheses)
        
        metrics_dict = {"BLEU-1": b1, "BLEU-2": b2, "BLEU-3": b3, "BLEU-4": b4, "METEOR": m, "ROUGE-L": r}
        mlflow.log_metrics(metrics_dict)
        print(f"\n📊 FINAL RESULTS ({eval_limit} SAMPLES):\n{json.dumps(metrics_dict, indent=2)}")

        with open(f"results/{model_id}_summary.json", 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        mlflow.log_artifact(f"results/{model_id}_summary.json")

if __name__ == "__main__":
    main()
import os
import tensorflow as tf
import numpy as np
import mlflow
import dagshub
import cv2
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient
from src.utils.config_loader import load_config
from src.data.dataset_loader import DataLoader
from src.models.encoder import CNN_Encoder
from src.models.decoder import RNN_Decoder
from src.utils.metrics import calculate_all_metrics

def generate_caption_smart(image_tensor, encoder, decoder, text_processor, config):
    """Generates a caption and records the raw attention weights for visualization."""
    features = encoder(image_tensor)
    num_features = features.shape[1] 
    hidden = decoder.init_decoder_state(features)
    start_token = text_processor.tokenizer.word_index['<start>']
    dec_input = tf.expand_dims([start_token], 0)
    
    result = []
    # Store attention weights: (Max_Length, Number_of_Pixels)
    attention_plot = np.zeros((config['dataset']['max_caption_length'], num_features))
    used_ids = [] # To prevent word repetition

    for i in range(config['dataset']['max_caption_length']):
        preds, hidden, attn_weights = decoder(dec_input, features, hidden)
        
        if attn_weights is not None:
            # Flatten weights to 1D array per word
            attention_plot[i] = tf.reshape(attn_weights, (-1,)).numpy()

        logits = preds[0].numpy()
        # Apply a small penalty to words already used to encourage diversity
        for w_id in used_ids:
            logits[w_id] -= 2.0 

        predicted_id = np.argmax(logits)
        word = text_processor.tokenizer.index_word.get(predicted_id, '<unk>')
        
        if word == '<end>':
            break
        
        result.append(word)
        used_ids.append(predicted_id)
        dec_input = tf.expand_dims([predicted_id], 0)
        
    return ' '.join(result), attention_plot

def plot_attention_grid(image, result, attention_plot, sample_idx, model_id):
    """
    Creates a PhD-level heatmap grid. 
    Uses Gaussian Blurring and Normalization to make objects 'glow' instead of 'pixels'.
    """
    words = result.split()
    # Calculate grid size (e.g., 10 for Xception's 100 features)
    grid_size = int(np.sqrt(attention_plot.shape[1]))
    
    num_words = len(words)
    cols = 3
    rows = (num_words // cols) + (1 if num_words % cols != 0 else 0)
    
    fig = plt.figure(figsize=(cols * 5, rows * 5))
    
    for i in range(num_words):
        # 1. Reshape the 1D weights back to a 2D grid (e.g., 10x10)
        att_map = attention_plot[i].reshape(grid_size, grid_size)
        
        # 2. PRO-SMOOTHING: Resize to image dimensions using Bicubic Interpolation
        # This turns the 10x10 blocks into a smooth 2D field
        att_map_resized = cv2.resize(att_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
        
        # 3. GAUSSIAN BLUR: Softens the focus so it looks like a professional heatmap
        att_map_resized = cv2.GaussianBlur(att_map_resized, (21, 21), 0)
        
        # 4. MIN-MAX NORMALIZATION: Ensures the focus for EVERY word is clearly visible
        if np.max(att_map_resized) > 0:
            att_map_resized = att_map_resized / np.max(att_map_resized)

        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(f"Focus: {words[i]}", fontsize=16, fontweight='bold')
        
        # Plot base image
        ax.imshow(image)
        # Overlay heatmap with 'jet' (classic blue-to-red scale)
        # alpha=0.45 makes it perfectly transparent
        ax.imshow(att_map_resized, cmap='jet', alpha=0.45) 
        ax.axis('off')

    plt.tight_layout()
    path = f"results/samples/{model_id}_Heatmap_{sample_idx}.png"
    os.makedirs("results/samples", exist_ok=True)
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()
    return path

def main():
    config = load_config()
    
    # --- METADATA RECONSTRUCTION ---
    enc, dec = config['model']['encoder_name'], config['model']['decoder_type']
    layers, units = config['model']['num_layers'], config['model']['units']
    subset, epochs = config['dataset']['subset_size'], config['training']['epochs']
    attn = config['model'].get('attention_type', 'None')
    opt = "GWO" if config['training'].get('optimizer_type') == "greywolf" else "Adam"
    tf_val = "TF" if config['training'].get('use_teacher_forcing', False) else "Base"
    
    model_id = f"ENC_{enc}_DEC_{dec}_L{layers}_U{units}_S{subset}_E{epochs}_{tf_val}_{attn}_{opt}"
    
    # Initialize MLflow & DagsHub
    dagshub.init(repo_owner=config['mlflow']['repo_owner'], repo_name=config['mlflow']['repo_name'], mlflow=True)
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(model_id)

    # 1. Load Data & Tokenizer
    loader = DataLoader(config)
    img_paths, captions = loader.load_annotations()
    (_, _), (_, _), (test_imgs, test_caps) = loader.split_data(img_paths, captions)
    loader.text_processor.fit_on_texts(captions) # Ensure vocabulary is identical

    # 2. Build Models
    encoder, decoder = CNN_Encoder(config), RNN_Decoder(config)
    
    # Dry run to initialize shapes
    encoder(tf.zeros((1, 299, 299, 3)))
    num_f = 100 if enc == "Xception" else 64
    decoder(tf.zeros((1, 1)), tf.zeros((1, num_f, units)), decoder.init_decoder_state(tf.zeros((1, num_f, units))))

    # 3. Load Trained Weights
    ckpt_dir = config['training']['checkpoint_path']
    enc_path = os.path.join(ckpt_dir, f"{model_id}_enc.weights.h5")
    dec_path = os.path.join(ckpt_dir, f"{model_id}_dec.weights.h5")
    
    if os.path.exists(enc_path):
        encoder.load_weights(enc_path)
        decoder.load_weights(dec_path)
        print(f"✅ Loaded weights: {model_id}")
    else:
        print(f"❌ Weights not found for {model_id}. Run training first!"); return

    # 4. Evaluation Loop
    b1_l, b2_l, b3_l, b4_l, met_l, rou_l = [], [], [], [], [], []
    
    with mlflow.start_run(run_name="Final_Evaluation"):
        os.makedirs("results/samples", exist_ok=True)
        print(f"🚀 Starting final evaluation for {len(test_imgs)} test images...")
        
        for i in range(min(50, len(test_imgs))): # Evaluate 50 samples
            # Preprocess image
            img_tensor, _ = loader.image_processor.preprocess_image(test_imgs[i])
            
            # Generate Caption and Attention Map
            pred, attn_plot = generate_caption_smart(tf.expand_dims(img_tensor, 0), encoder, decoder, loader.text_processor, config)
            
            # Calculate Scores
            (b1, b2, b3, b4), m, r = calculate_all_metrics(test_caps[i], pred)
            b1_l.append(b1); b2_l.append(b2); b3_l.append(b3); b4_l.append(b4); met_l.append(m); rou_l.append(r)

            # Log first 5 samples as Visual Artifacts
            if i < 5:
                # Denormalize image for display (Xception is [-1, 1] -> [0, 1])
                display_img = (img_tensor.numpy() + 1.0) / 2.0
                display_img = np.clip(display_img, 0, 1)

                # 1. Save standard side-by-side comparison
                plt.figure(figsize=(10, 8))
                plt.imshow(display_img)
                c_real = test_caps[i].replace('<start>', '').replace('<end>', '').strip()
                plt.title(f"REAL: {c_real}\nPRED: {pred}\nBLEU-4: {b4:.4f}", fontsize=10)
                plt.axis('off')
                path_std = f"results/samples/Sample_{i}_Result.png"
                plt.savefig(path_std); plt.close()
                mlflow.log_artifact(path_std)

                # 2. Save the Professional Heatmap Grid
                if attn != 'None':
                    heatmap_path = plot_attention_grid(display_img, pred, attn_plot, i, model_id)
                    mlflow.log_artifact(heatmap_path)

        # 5. Log Summary Metrics
        summary = {
            "test_bleu1": np.mean(b1_l), "test_bleu2": np.mean(b2_l),
            "test_bleu3": np.mean(b3_l), "test_bleu4": np.mean(b4_l),
            "test_meteor": np.mean(met_l), "test_rougeL": np.mean(rou_l)
        }
        mlflow.log_metrics(summary)
        print(f"📊 Final Results - BLEU-4: {summary['test_bleu4']:.4f} | METEOR: {summary['test_meteor']:.4f}")

if __name__ == "__main__":
    main()
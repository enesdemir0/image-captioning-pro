import os
import tensorflow as tf
import numpy as np
import mlflow
import dagshub
import cv2
import matplotlib.pyplot as plt
from src.utils.config_loader import load_config
from src.data.dataset_loader import DataLoader
from src.models.encoder import CNN_Encoder
from src.models.decoder import RNN_Decoder
from src.utils.metrics import calculate_all_metrics

def generate_caption_smart(image_tensor, encoder, decoder, text_processor, config):
    features = encoder(image_tensor)
    num_features = features.shape[1] 
    hidden = decoder.init_decoder_state(features)
    start_token = text_processor.tokenizer.word_index['<start>']
    dec_input = tf.expand_dims([start_token], 0)
    
    result = []
    attention_plot = np.zeros((config['dataset']['max_caption_length'], num_features))
    used_ids = [] 

    for i in range(config['dataset']['max_caption_length']):
        preds, hidden, attn_weights = decoder(dec_input, features, hidden)
        if attn_weights is not None:
            attention_plot[i] = tf.reshape(attn_weights, (-1,)).numpy()

        logits = preds[0].numpy()
        for w_id in used_ids:
            logits[w_id] -= 2.0 

        predicted_id = np.argmax(logits)
        word = text_processor.tokenizer.index_word.get(predicted_id, '<unk>')
        if word == '<end>': break
        
        result.append(word)
        used_ids.append(predicted_id)
        dec_input = tf.expand_dims([predicted_id], 0)
        
    return ' '.join(result), attention_plot

def plot_attention_grid(image, result, attention_plot, sample_idx, model_id):
    words = result.split()
    grid_size = int(np.sqrt(attention_plot.shape[1]))
    num_words = len(words)
    cols = 3
    rows = (num_words // cols) + (1 if num_words % cols != 0 else 0)
    
    fig = plt.figure(figsize=(cols * 5, rows * 5))
    for i in range(num_words):
        att_map = attention_plot[i].reshape(grid_size, grid_size)
        att_map_resized = cv2.resize(att_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
        att_map_resized = cv2.GaussianBlur(att_map_resized, (21, 21), 0)
        
        if np.max(att_map_resized) > 0:
            att_map_resized = att_map_resized / np.max(att_map_resized)

        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(f"Focus: {words[i]}", fontsize=16, fontweight='bold')
        ax.imshow(image)
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
    
    # --- MODEL ID CONSTRUCTION (Matches Debug Script) ---
    enc, dec = config['model']['encoder_name'], config['model']['decoder_type']
    layers, units = config['model']['num_layers'], config['model']['units']
    subset, epochs = config['dataset']['subset_size'], config['training']['epochs']
    attn = config['model'].get('attention_type', 'None')
    opt = "Adam" if config['training'].get('optimizer_type') == "adam" else "GWO"
    tf_val = "TF" if config['training']['use_teacher_forcing'] else "Base"
    
    model_id = f"ENC_{enc}_DEC_{dec}_L{layers}_U{units}_S{subset}_E{epochs}_{tf_val}_{attn}_{opt}"
    print(f"🧐 Evaluating: {model_id}")

    # MLflow Setup
    dagshub.init(repo_owner="enesdemir0", repo_name="image-captioning-pro", mlflow=True)
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(model_id)

    # 1. Load Data
    loader = DataLoader(config)
    img_paths, captions = loader.load_annotations()
    (_, _), (_, _), (test_imgs, test_caps) = loader.split_data(img_paths, captions)
    loader.text_processor.fit_on_texts(captions)

    # 2. Build Model
    encoder, decoder = CNN_Encoder(config), RNN_Decoder(config)
    encoder(tf.zeros((1, 299, 299, 3)))
    num_f = 100 if enc == "Xception" else 64
    decoder(tf.zeros((1, 1)), tf.zeros((1, num_f, units)), decoder.init_decoder_state(tf.zeros((1, units))))

    # 3. LOAD WEIGHTS (The Fixed Part)
    ckpt_dir = config['training']['checkpoint_path']
    e_path = os.path.join(ckpt_dir, f"{model_id}_encoder.weights.h5")
    d_path = os.path.join(ckpt_dir, f"{model_id}_decoder.weights.h5")
    
    if os.path.exists(e_path):
        encoder.load_weights(e_path)
        decoder.load_weights(d_path)
        print(f"✅ SUCCESS: Loaded weights from {e_path}")
    else:
        print(f"❌ ERROR: Weights not found at {e_path}")
        return

    # 4. Run Evaluation
    b1_l, b2_l, b3_l, b4_l, met_l, rou_l = [], [], [], [], [], []
    with mlflow.start_run(run_name="Professional_Evaluation"):
        for i in range(min(10, len(test_imgs))): # Evaluating first 10
            img_tensor, _ = loader.image_processor.preprocess_image(test_imgs[i])
            pred, attn_plot = generate_caption_smart(tf.expand_dims(img_tensor, 0), encoder, decoder, loader.text_processor, config)
            
            (b1, b2, b3, b4), m, r = calculate_all_metrics(test_caps[i], pred)
            b1_l.append(b1); b2_l.append(b2); b3_l.append(b3); b4_l.append(b4); met_l.append(m); rou_l.append(r)

            if i < 5:
                # Denormalize
                display_img = (img_tensor.numpy() + 1.0) / 2.0
                display_img = np.clip(display_img, 0, 1)
                
                # Result Plot
                plt.figure(figsize=(10, 8)); plt.imshow(display_img)
                c_real = test_caps[i].replace('<start>', '').replace('<end>', '').strip()
                plt.title(f"REAL: {c_real}\nPRED: {pred}\nBLEU-4: {b4:.4f}", fontsize=10)
                plt.axis('off'); path_std = f"results/samples/Sample_{i}_Result.png"
                plt.savefig(path_std); plt.close(); mlflow.log_artifact(path_std)

                # Heatmap Grid
                if attn != 'None':
                    heatmap_path = plot_attention_grid(display_img, pred, attn_plot, i, model_id)
                    mlflow.log_artifact(heatmap_path)

        summary = {"test_bleu4": np.mean(b4_l), "test_meteor": np.mean(met_l)}
        mlflow.log_metrics(summary)
        print(f"📊 Results - BLEU-4: {summary['test_bleu4']:.4f} | METEOR: {summary['test_meteor']:.4f}")

if __name__ == "__main__":
    main()
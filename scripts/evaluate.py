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
    hidden = decoder.init_decoder_state(features)
    start_token = text_processor.tokenizer.word_index['<start>']
    dec_input = tf.expand_dims([start_token], 0)
    result = []
    attention_plot = np.zeros((config['dataset']['max_caption_length'], features.shape[1]))
    for i in range(config['dataset']['max_caption_length']):
        preds, hidden, attn_weights = decoder(dec_input, features, hidden)
        if attn_weights is not None: attention_plot[i] = tf.reshape(attn_weights, (-1,)).numpy()
        predicted_id = np.argmax(preds[0])
        word = text_processor.tokenizer.index_word.get(predicted_id, '<unk>')
        if word == '<end>': break
        result.append(word)
        dec_input = tf.expand_dims([predicted_id], 0)
    return ' '.join(result), attention_plot

def plot_attention_grid(image, result, attention_plot, sample_idx, model_id):
    words = result.split()
    grid_size = int(np.sqrt(attention_plot.shape[1]))
    cols = 3
    rows = (len(words) // cols) + (1 if len(words) % cols != 0 else 0)
    fig = plt.figure(figsize=(cols * 5, rows * 5))
    for i in range(len(words)):
        att_map = cv2.resize(attention_plot[i].reshape(grid_size, grid_size), (image.shape[1], image.shape[0]), interpolation=cv2.INTER_CUBIC)
        att_map = cv2.GaussianBlur(att_map, (21, 21), 0)
        if np.max(att_map) > 0: att_map /= np.max(att_map)
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_title(f"Focus: {words[i]}", fontsize=16, fontweight='bold')
        ax.imshow(image); ax.imshow(att_map, cmap='jet', alpha=0.45); ax.axis('off')
    path = f"results/samples/{model_id}_Heatmap_{sample_idx}.png"
    os.makedirs("results/samples", exist_ok=True)
    plt.savefig(path, bbox_inches='tight', dpi=150); plt.close()
    return path

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

    loader = DataLoader(config)
    img_paths, captions = loader.load_annotations()
    (tr_i, tr_c), _, (test_i, test_c) = loader.split_data(img_paths, captions)
    loader.text_processor.fit_on_texts(tr_c)

    encoder, decoder = CNN_Encoder(config), RNN_Decoder(config)
    encoder(tf.zeros((1, 299, 299, 3)))
    num_f = 100 if enc == "Xception" else 64
    decoder(tf.zeros((1, 1)), tf.zeros((1, num_f, units)), decoder.init_decoder_state(tf.zeros((1, num_f, units))))

    # --- SMART SEARCH: Check Local FIRST, then Drive ---
    drive_path = config['training']['checkpoint_path']
    local_path = "./checkpoints/"
    e_path = os.path.join(local_path, f"{model_id}_encoder.weights.h5")
    d_path = os.path.join(local_path, f"{model_id}_decoder.weights.h5")

    if not os.path.exists(e_path):
        print("ℹ️ Local weights not found, checking Drive...")
        e_path = os.path.join(drive_path, f"{model_id}_encoder.weights.h5")
        d_path = os.path.join(drive_path, f"{model_id}_decoder.weights.h5")

    if os.path.exists(e_path):
        print(f"✅ Loading: {e_path}")
        encoder.load_weights(e_path); decoder.load_weights(d_path)
    else:
        print(f"❌ ERROR: Weights not found in local session or Drive!"); return

    b1_l, b2_l, b3_l, b4_l, met_l, rou_l = [], [], [], [], [], []
    with mlflow.start_run(run_name="Final_Test_Set_Evaluation"):
        for i in range(len(test_i)):
            img_tensor, _ = loader.image_processor.preprocess_image(test_i[i])
            pred, attn_plot = generate_caption_smart(tf.expand_dims(img_tensor, 0), encoder, decoder, loader.text_processor, config)
            (b1, b2, b3, b4), m, r = calculate_all_metrics(test_c[i], pred)
            b1_l.append(b1); b2_l.append(b2); b3_l.append(b3); b4_l.append(b4); met_l.append(m); rou_l.append(r)
            if i < 10:
                display_img = np.clip((img_tensor.numpy() + 1.0) / 2.0, 0, 1)
                plt.figure(figsize=(10, 8)); plt.imshow(display_img)
                plt.title(f"REAL: {test_c[i]}\nPRED: {pred}\nB4: {b4:.4f}", fontsize=10); plt.axis('off')
                path_std = f"results/samples/Sample_{i}_Result.png"; plt.savefig(path_std); plt.close()
                mlflow.log_artifact(path_std)
                if attn != 'None': mlflow.log_artifact(plot_attention_grid(display_img, pred, attn_plot, i, model_id))
            if i % 100 == 0: print(f"📊 Progress: {i}/{len(test_i)}...")
        summary = {"test_bleu4": np.mean(b4_l), "test_meteor": np.mean(met_l), "test_rougeL": np.mean(rou_l)}
        mlflow.log_metrics(summary)
        print(f"📊 FINAL RESULTS: BLEU-4: {summary['test_bleu4']:.4f}")

if __name__ == "__main__":
    main()
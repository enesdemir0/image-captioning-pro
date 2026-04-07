import os
import tensorflow as tf
import numpy as np
import mlflow
import dagshub
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient
from src.utils.config_loader import load_config
from src.data.dataset_loader import DataLoader
from src.models.encoder import CNN_Encoder
from src.models.decoder import RNN_Decoder
from src.utils.metrics import calculate_all_metrics

def generate_caption_smart(image_tensor, encoder, decoder, text_processor, config):
    features = encoder(image_tensor)
    num_features = features.shape[1] 
    hidden = decoder.init_decoder_state(tf.reduce_mean(features, axis=1))
    start_token = text_processor.tokenizer.word_index['<start>']
    dec_input = tf.expand_dims([start_token], 0)
    
    result = []
    attention_plot = np.zeros((config['dataset']['max_caption_length'], num_features))
    used_ids = [] # For repetition penalty

    for i in range(config['dataset']['max_caption_length']):
        preds, hidden, attn_weights = decoder(dec_input, features, hidden)
        if attn_weights is not None:
            attention_plot[i] = tf.reshape(attn_weights, (-1,)).numpy()

        logits = preds[0].numpy()
        # Repetition Penalty: Lower probability of words already used
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
    temp_image = np.array(image)
    words = result.split()
    grid_size = int(np.sqrt(attention_plot.shape[1]))
    fig = plt.figure(figsize=(12, 12))
    for i in range(len(words)):
        att_map = np.resize(attention_plot[i], (grid_size, grid_size))
        ax = fig.add_subplot(len(words) // 3 + 1, 3, i + 1)
        ax.set_title(words[i], fontsize=12)
        plt.imshow(temp_image)
        plt.imshow(att_map, cmap='gray', alpha=0.5, extent=plt.gca().get_extent(), interpolation='bilinear')
        plt.axis('off')
    plt.tight_layout()
    path = f"results/samples/{model_id}_Heatmap_{sample_idx}.png"
    plt.savefig(path); plt.close(); return path

def main():
    config = load_config()
    
    # 1. Build Model ID Dynamically from YAML
    enc, dec = config['model']['encoder_name'], config['model']['decoder_type']
    layers, units = config['model']['num_layers'], config['model']['units']
    subset, epochs = config['dataset']['subset_size'], config['training']['epochs']
    attn = config['model'].get('attention_type', 'None')
    opt = "GWO" if config['training'].get('optimizer_type') == "greywolf" else "Adam"
    tf_val = "TF" if config['training'].get('use_teacher_forcing', False) else "Base"
    
    model_id = f"ENC_{enc}_DEC_{dec}_L{layers}_U{units}_S{subset}_E{epochs}_{tf_val}_{attn}_{opt}"
    
    dagshub.init(repo_owner="enesdemir0", repo_name="image-captioning-pro", mlflow=True)
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    
    client = MlflowClient()
    exp = client.get_experiment_by_name(model_id)
    if exp and exp.lifecycle_stage == 'deleted': client.restore_experiment(exp.experiment_id)
    mlflow.set_experiment(model_id)

    # 2. Data Setup
    loader = DataLoader(config)
    img_paths, captions = loader.load_annotations()
    if subset > 0: img_paths, captions = img_paths[:subset], captions[:subset]
    (_, _), (_, _), (test_imgs, test_caps) = loader.split_data(img_paths, captions)
    loader.text_processor.fit_on_texts(captions)

    # 3. Model Init
    encoder, decoder = CNN_Encoder(config), RNN_Decoder(config)
    encoder(tf.zeros((1, 299, 299, 3)))
    num_f = 100 if enc == "Xception" else 64
    decoder(tf.zeros((1, 1)), tf.zeros((1, num_f, units)), decoder.init_decoder_state(tf.zeros((1, units))))

    # 4. Load Weights (Try both naming styles)
    ckpt_dir = config['training']['checkpoint_path']
    # Try style 1 (_enc) then style 2 (_encoder)
    possible_enc = [f"{model_id}_enc.weights.h5", f"{model_id}_encoder.weights.h5"]
    possible_dec = [f"{model_id}_dec.weights.h5", f"{model_id}_decoder.weights.h5"]
    
    loaded = False
    for e_file, d_file in zip(possible_enc, possible_dec):
        e_path = os.path.join(ckpt_dir, e_file)
        d_path = os.path.join(ckpt_dir, d_file)
        if os.path.exists(e_path):
            encoder.load_weights(e_path); decoder.load_weights(d_path)
            print(f"✅ Successfully loaded weights: {e_file}")
            loaded = True; break
    
    if not loaded:
        print(f"❌ Error: Could not find weights for {model_id} on Drive."); return

    # 5. Evaluation Loop
    b1_l, b2_l, b3_l, b4_l, met_l, rou_l = [], [], [], [], [], []
    with mlflow.start_run(run_name="Final_Evaluation"):
        os.makedirs("results/samples", exist_ok=True)
        print(f"--- Evaluating {model_id} ---")
        for i in range(min(50, len(test_imgs))):
            img_tensor, _ = loader.image_processor.preprocess_image(test_imgs[i])
            pred, attn_plot = generate_caption_smart(tf.expand_dims(img_tensor, 0), encoder, decoder, loader.text_processor, config)
            
            (b1, b2, b3, b4), m, r = calculate_all_metrics(test_caps[i], pred)
            b1_l.append(b1); b2_l.append(b2); b3_l.append(b3); b4_l.append(b4); met_l.append(m); rou_l.append(r)

            if i < 5:
                # Save visual sample
                plt.figure(figsize=(10, 10)); plt.imshow(img_tensor.numpy() * 0.5 + 0.5)
                c_real = test_caps[i].replace('<start>', '').replace('<end>', '').strip()
                plt.title(f"REAL: {c_real}\nPRED: {pred}\nBLEU-4: {b4:.4f}", fontsize=10)
                plt.axis('off'); path = f"results/samples/Sample_{i}.png"
                plt.savefig(path); plt.close(); mlflow.log_artifact(path)
                if attn != 'None':
                    mlflow.log_artifact(plot_attention_grid(img_tensor.numpy() * 0.5 + 0.5, pred, attn_plot, i, model_id))

        mlflow.log_metrics({"test_bleu1": np.mean(b1_l), "test_bleu2": np.mean(b2_l), "test_bleu3": np.mean(b3_l), "test_bleu4": np.mean(b4_l), "test_meteor": np.mean(met_l), "test_rougeL": np.mean(rou_l)})
        print(f"Final Average BLEU-4: {np.mean(b4_l):.4f}")

if __name__ == "__main__":
    main()
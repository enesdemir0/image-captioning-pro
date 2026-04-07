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

def generate_caption_beam(image_tensor, encoder, decoder, text_processor, config, beam_index=3):
    start_token = [text_processor.tokenizer.word_index['<start>']]
    beam = [[start_token, 0.0]]
    features = encoder(image_tensor)
    hidden = decoder.init_decoder_state(tf.reduce_mean(features, axis=1))
    for i in range(config['dataset']['max_caption_length']):
        candidates = []
        for s in beam:
            dec_input = tf.expand_dims([s[0][-1]], 0)
            preds, hidden, _ = decoder(dec_input, features, hidden)
            top_preds = tf.math.top_k(preds[0], k=beam_index)
            for j in range(beam_index):
                word_id = top_preds.indices[j].numpy()
                candidates.append([s[0] + [word_id], s[1] + top_preds.values[j].numpy()])
        beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_index]
        if text_processor.tokenizer.index_word.get(beam[0][0][-1]) == '<end>': break
    return ' '.join([text_processor.tokenizer.index_word.get(i, '<unk>') for i in beam[0][0]][1:-1])

def generate_caption_greedy(image_tensor, encoder, decoder, text_processor, config):
    features = encoder(image_tensor)
    hidden = decoder.init_decoder_state(tf.reduce_mean(features, axis=1))
    dec_input = tf.expand_dims([text_processor.tokenizer.word_index['<start>']], 0)
    result, attention_plot = [], np.zeros((config['dataset']['max_caption_length'], features.shape[1]))
    for i in range(config['dataset']['max_caption_length']):
        preds, hidden, attn_weights = decoder(dec_input, features, hidden)
        if attn_weights is not None: attention_plot[i] = tf.reshape(attn_weights, (-1,)).numpy()
        predicted_id = tf.argmax(preds[0]).numpy()
        word = text_processor.tokenizer.index_word.get(predicted_id, '<unk>')
        if word == '<end>': break
        result.append(word); dec_input = tf.expand_dims([predicted_id], 0)
    return ' '.join(result), attention_plot

def plot_attention_grid(image, result, attention_plot, sample_idx, model_id):
    temp_image = np.array(image); words = result.split()
    grid_size = int(np.sqrt(attention_plot.shape[1]))
    fig = plt.figure(figsize=(12, 12))
    for i in range(len(words)):
        att_map = np.resize(attention_plot[i], (grid_size, grid_size))
        ax = fig.add_subplot(len(words) // 3 + 1, 3, i + 1)
        ax.set_title(words[i]); img = ax.imshow(temp_image)
        ax.imshow(att_map, cmap='gray', alpha=0.6, extent=img.get_extent(), interpolation='bilinear')
        ax.axis('off')
    plt.tight_layout(); path = f"results/samples/{model_id}_Heatmap_{sample_idx}.png"
    plt.savefig(path); plt.close(); return path

def main():
    config = load_config()
    enc, dec, layers, units, subset, epochs = config['model']['encoder_name'], config['model']['decoder_type'], config['model']['num_layers'], config['model']['units'], config['dataset']['subset_size'], config['training']['epochs']
    attn = config['model'].get('attention_type', 'None')
    opt = "GWO" if config['training'].get('optimizer_type') == "greywolf" else "Adam"
    tf_val = "TF" if config['training']['use_teacher_forcing'] else "Base"
    model_id = f"ENC_{enc}_DEC_{dec}_L{layers}_U{units}_S{subset}_E{epochs}_{tf_val}_{attn}_{opt}"
    
    dagshub.init(repo_owner=config['mlflow']['repo_owner'], repo_name=config['mlflow']['repo_name'], mlflow=True)
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri']); mlflow.set_experiment(model_id)

    loader = DataLoader(config)
    img_paths, captions = loader.load_annotations()
    if subset > 0: img_paths, captions = img_paths[:subset], captions[:subset]
    _, _, (test_imgs, test_caps) = loader.split_data(img_paths, captions)
    loader.text_processor.fit_on_texts(captions)

    encoder, decoder = CNN_Encoder(config), RNN_Decoder(config)
    encoder(tf.zeros((1, 299, 299, 3)))
    num_f = 100 if enc == "Xception" else 64
    decoder(tf.zeros((1, 1)), tf.zeros((1, num_f, units)), decoder.init_decoder_state(tf.zeros((1, units))))

    ckpt_dir = config['training']['checkpoint_path']
    encoder.load_weights(os.path.join(ckpt_dir, f"{model_id}_enc.weights.h5"))
    decoder.load_weights(os.path.join(ckpt_dir, f"{model_id}_dec.weights.h5"))

    # --- LISTS FOR ALL 6 METRICS ---
    b1_l, b2_l, b3_l, b4_l, met_l, rou_l = [], [], [], [], [], []

    with mlflow.start_run(run_name="Evaluation"):
        os.makedirs("results/samples", exist_ok=True)
        for i in range(min(50, len(test_imgs))):
            img_tensor, _ = loader.image_processor.preprocess_image(test_imgs[i])
            pred_beam = generate_caption_beam(tf.expand_dims(img_tensor, 0), encoder, decoder, loader.text_processor, config)
            
            # UNPACK ALL BLEU SCORES
            (b1, b2, b3, b4), m, r = calculate_all_metrics(test_caps[i], pred_beam)
            b1_l.append(b1); b2_l.append(b2); b3_l.append(b3); b4_l.append(b4)
            met_l.append(m); rou_l.append(r)

            if i < 5:
                plt.figure(figsize=(10, 10)); display_img = img_tensor.numpy() * 0.5 + 0.5
                plt.imshow(display_img); c_real = test_caps[i].replace('<start>', '').replace('<end>', '').strip()
                plt.title(f"REAL: {c_real}\nPRED: {pred_beam}", fontsize=10); plt.axis('off')
                path = f"results/samples/Sample_{i}.png"; plt.savefig(path); plt.close(); mlflow.log_artifact(path)
                if attn != 'None':
                    pg, ap = generate_caption_greedy(tf.expand_dims(img_tensor, 0), encoder, decoder, loader.text_processor, config)
                    mlflow.log_artifact(plot_attention_grid(display_img, pg, ap, i, model_id))

        # LOG ALL 6 METRICS
        mlflow.log_metrics({
            "test_bleu1": np.mean(b1_l), "test_bleu2": np.mean(b2_l),
            "test_bleu3": np.mean(b3_l), "test_bleu4": np.mean(b4_l),
            "test_meteor": np.mean(met_l), "test_rougeL": np.mean(rou_l)
        })
        print(f"Table row ready for DagsHub!")

if __name__ == "__main__":
    main()
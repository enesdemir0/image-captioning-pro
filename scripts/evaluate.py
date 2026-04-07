import os
import tensorflow as tf
import numpy as np
import mlflow
import dagshub
import matplotlib.pyplot as plt
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
                prob = top_preds.values[j].numpy()
                candidates.append([s[0] + [word_id], s[1] + prob])
        
        beam = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_index]
        if text_processor.tokenizer.index_word.get(beam[0][0][-1]) == '<end>': break

    best_path = beam[0][0]
    final_caption = [text_processor.tokenizer.index_word.get(i, '<unk>') for i in best_path]
    return ' '.join(final_caption[1:-1])

def main():
    config = load_config()
    enc_name = config['model']['encoder_name']
    dec_type = config['model']['decoder_type']
    layers = config['model']['num_layers']
    subset = config['dataset']['subset_size']
    epochs = config['training']['epochs']
    attn = config['model'].get('attention_type', 'None')
    
    # Matching the Naming
    if attn == "ran": phase = "Phase3_Bonus2"
    elif attn != "None": phase = "Phase3"
    else: phase = "Phase2" if config['training']['use_teacher_forcing'] else "Phase1"
    
    model_id = f"{phase}_ENC_{enc_name}_DEC_{dec_type}_L{layers}_S{subset}_E{epochs}_{attn}"
    
    dagshub.init(repo_owner=config['mlflow']['repo_owner'], repo_name=config['mlflow']['repo_name'], mlflow=True)
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(model_id)

    loader = DataLoader(config)
    img_paths, captions = loader.load_annotations()
    if subset > 0: img_paths, captions = img_paths[:subset], captions[:subset]
    (train_imgs, train_caps), _, (test_imgs, test_caps) = loader.split_data(img_paths, captions)
    loader.text_processor.fit_on_texts(train_caps)

    encoder, decoder = CNN_Encoder(config), RNN_Decoder(config)
    encoder(tf.zeros((1, 299, 299, 3)))
    num_feat = 100 if enc_name == "Xception" else 64
    decoder(tf.zeros((1, 1)), tf.zeros((1, num_feat, config['model']['units'])), decoder.init_decoder_state(tf.zeros((1, config['model']['units']))))

    ckpt_dir = config['training']['checkpoint_path']
    encoder.load_weights(os.path.join(ckpt_dir, f"{model_id}_encoder.weights.h5"))
    decoder.load_weights(os.path.join(ckpt_dir, f"{model_id}_decoder.weights.h5"))

    b1_l, b2_l, b3_l, b4_l, met_l, rou_l = [], [], [], [], [], []

    with mlflow.start_run(run_name="Evaluation_Final"):
        os.makedirs("results/samples", exist_ok=True)
        print(f"--- Generating Results for {model_id} ---")

        for i in range(min(50, len(test_imgs))):
            img_tensor, _ = loader.image_processor.preprocess_image(test_imgs[i])
            pred = generate_caption_beam(tf.expand_dims(img_tensor, 0), encoder, decoder, loader.text_processor, config)
            
            (b1, b2, b3, b4), m, r = calculate_all_metrics(test_caps[i], pred)
            b1_l.append(b1); b2_l.append(b2); b3_l.append(b3); b4_l.append(b4)
            met_l.append(m); rou_l.append(r)

            if i < 10:
                plt.figure(figsize=(10, 10))
                plt.imshow(img_tensor.numpy() * 0.5 + 0.5)
                
                # THE FIX: Clear labels for Real and Pred
                clean_real = test_caps[i].replace('<start>', '').replace('<end>', '').strip()
                plt.title(f"REAL: {clean_real}\nPRED: {pred}\nBLEU-4: {b4:.4f}", fontsize=12, color='blue', pad=20)
                plt.axis('off')
                
                fig_path = f"results/samples/Test_Result_{i}.png"
                plt.savefig(fig_path, bbox_inches='tight')
                plt.close()
                mlflow.log_artifact(fig_path)

        summary = {"test_bleu1": np.mean(b1_l), "test_bleu4": np.mean(b4_l), "test_meteor": np.mean(met_l), "test_rougeL": np.mean(rou_l)}
        mlflow.log_metrics(summary)
        print(f"\nFinal BLEU-4: {summary['test_bleu4']:.4f}")

if __name__ == "__main__":
    main()
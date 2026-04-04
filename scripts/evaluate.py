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

def generate_caption(image_tensor, encoder, decoder, text_processor, config):
    features = encoder(image_tensor)
    hidden = decoder.init_decoder_state(tf.reduce_mean(features, axis=1))
    start_token = text_processor.tokenizer.word_index['<start>']
    dec_input = tf.expand_dims([start_token], 0)
    
    result = []
    for i in range(config['dataset']['max_caption_length']):
        preds, hidden = decoder(dec_input, hidden)
        predicted_id = tf.argmax(preds[0]).numpy()
        word = text_processor.tokenizer.index_word.get(predicted_id, '<unk>')
        if word == '<end>': break
        result.append(word)
        dec_input = tf.expand_dims([predicted_id], 0)
    return ' '.join(result)

def main():
    config = load_config()
    
    # Unified Naming Convention
    enc = config['model']['encoder_name']
    dec = config['model']['decoder_type']
    layers = config['model']['num_layers']
    subset = config['dataset']['subset_size']
    epochs = config['training']['epochs']
    
    # Format: ENC_InceptionV3_DEC_GRU_L3_S20000_E30
    model_id = f"ENC_{enc}_DEC_{dec}_L{layers}_S{subset}_E{epochs}"
    
    dagshub.init(repo_owner=config['mlflow']['repo_owner'], repo_name=config['mlflow']['repo_name'], mlflow=True)
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(model_id)

    loader = DataLoader(config)
    img_paths, captions = loader.load_annotations()
    
    subset_size = config['dataset'].get('subset_size', 0)
    if subset_size > 0:
        img_paths, captions = img_paths[:subset_size], captions[:subset_size]

    (train_imgs, train_caps), _, (test_imgs, test_caps) = loader.split_data(img_paths, captions)
    loader.text_processor.fit_on_texts(train_caps)

    # Initialize and Force Build
    encoder, decoder = CNN_Encoder(config), RNN_Decoder(config)
    encoder(tf.zeros((1, 299, 299, 3)))
    decoder(tf.zeros((1, 1)), decoder.init_decoder_state(tf.zeros((1, config['model']['units']))))

    # Load Weights
    ckpt_dir = config['training']['checkpoint_path']
    encoder.load_weights(os.path.join(ckpt_dir, f"{model_id}_encoder.weights.h5"))
    decoder.load_weights(os.path.join(ckpt_dir, f"{model_id}_decoder.weights.h5"))

    # Metric Lists
    b1_l, b2_l, b3_l, b4_l, met_l, rou_l = [], [], [], [], [], []

    with mlflow.start_run(run_name="Evaluation_Phase"):
        print(f"\n--- Running Full Evaluation for {model_id} ---")
        os.makedirs("results/samples", exist_ok=True)

        for i in range(min(100, len(test_imgs))):
            img_tensor, _ = loader.image_processor.preprocess_image(test_imgs[i])
            pred = generate_caption(tf.expand_dims(img_tensor, 0), encoder, decoder, loader.text_processor, config)
            
            (b1, b2, b3, b4), m, r = calculate_all_metrics(test_caps[i], pred)
            
            b1_l.append(b1); b2_l.append(b2); b3_l.append(b3); b4_l.append(b4)
            met_l.append(m); rou_l.append(r)

            if i < 5:
                plt.figure(figsize=(10, 8))
                plt.imshow(img_tensor.numpy() * 0.5 + 0.5)
                c_real = test_caps[i].replace('<start>', '').replace('<end>', '').strip()
                c_pred = pred.replace('<start>', '').replace('<end>', '').strip()
                plt.title(f"REAL: {c_real}\nPRED: {c_pred}\nBLEU-4: {b4:.4f}")
                plt.axis('off')
                
                fig_name = f"Sample_{i}.png"
                plt.savefig(f"results/samples/{fig_name}")
                plt.close()
                mlflow.log_artifact(f"results/samples/{fig_name}")

        # Final Log to DagsHub
        summary = {
            "test_bleu1": np.mean(b1_l), "test_bleu2": np.mean(b2_l),
            "test_bleu3": np.mean(b3_l), "test_bleu4": np.mean(b4_l),
            "test_meteor": np.mean(met_l), "test_rougeL": np.mean(rou_l)
        }
        mlflow.log_metrics(summary)

        # Print Scientific Table
        print("\n" + "="*60)
        print(f"FINAL METRICS TABLE: {model_id}")
        print("-" * 60)
        print(f"BLEU-1: {summary['test_bleu1']:.4f} | BLEU-2: {summary['test_bleu2']:.4f}")
        print(f"BLEU-3: {summary['test_bleu3']:.4f} | BLEU-4: {summary['test_bleu4']:.4f}")
        print(f"METEOR: {summary['test_meteor']:.4f} | ROUGE-L: {summary['test_rougeL']:.4f}")
        print("="*60)

if __name__ == "__main__":
    main()
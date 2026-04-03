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
    model_id = f"{config['model']['encoder_name']}_{config['model']['decoder_type']}_L{config['model']['num_layers']}"
    
    dagshub.init(repo_owner=config['mlflow']['repo_owner'], repo_name=config['mlflow']['repo_name'], mlflow=True)
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])

    loader = DataLoader(config)
    img_paths, captions = loader.load_annotations()
    
    subset_size = config['dataset'].get('subset_size', 5000)
    img_paths, captions = img_paths[:subset_size], captions[:subset_size]
    (train_imgs, train_caps), _, (test_imgs, test_caps) = loader.split_data(img_paths, captions)

    loader.text_processor.fit_on_texts(train_caps)
    encoder, decoder = CNN_Encoder(config), RNN_Decoder(config)
    
    # Build models
    encoder(tf.zeros((1, 299, 299, 3)))
    dummy_state = decoder.init_decoder_state(tf.zeros((1, config['model']['units'])))
    decoder(tf.zeros((1, 1)), dummy_state)

    # Load Weights
    ckpt_dir = config['training']['checkpoint_path']
    encoder.load_weights(os.path.join(ckpt_dir, f"{model_id}_encoder.weights.h5"))
    decoder.load_weights(os.path.join(ckpt_dir, f"{model_id}_decoder.weights.h5"))

    bleus, meteors, rouges = [], [], []

    # Start a specific "Evaluation" run
    with mlflow.start_run(run_name=f"EVAL_{model_id}"):
        print(f"\n--- Generating Figures and Metrics for {model_id} ---")
        
        # Folder to store images locally before uploading to MLflow
        os.makedirs("results/samples", exist_ok=True)

        for i in range(50):
            img_tensor, _ = loader.image_processor.preprocess_image(test_imgs[i])
            pred = generate_caption(tf.expand_dims(img_tensor, 0), encoder, decoder, loader.text_processor, config)
            
            b, m, r = calculate_all_metrics(test_caps[i], pred)
            bleus.append(b); meteors.append(m); rouges.append(r)

            # --- CREATE FIGURE 5/6 STYLE ARTIFACTS ---
            if i < 5: # Save the first 5 test images as visual samples
                plt.figure(figsize=(10, 8))
                # De-normalize image for visualization (Inception scales to -1 to 1)
                display_img = img_tensor.numpy() * 0.5 + 0.5 
                plt.imshow(display_img)
                plt.title(f"REAL: {test_caps[i]}\nPRED: {pred}\nBLEU-4: {b:.4f}", fontsize=10)
                plt.axis('off')
                
                fig_path = f"results/samples/test_sample_{i}.png"
                plt.savefig(fig_path)
                plt.close()
                
                # LOG TO MLFLOW AS ARTIFACT
                mlflow.log_artifact(fig_path)
                print(f"Logged sample {i} to DagsHub")

        # Log Average Metrics
        metrics_summary = {
            "test_bleu4": np.mean(bleus),
            "test_meteor": np.mean(meteors),
            "test_rougeL": np.mean(rouges)
        }
        mlflow.log_metrics(metrics_summary)
        
        print("\n" + "="*40)
        print(f"FINAL TABLE RESULTS for {model_id}")
        print(f"BLEU-4: {metrics_summary['test_bleu4']:.4f}")
        print(f"METEOR: {metrics_summary['test_meteor']:.4f}")
        print(f"ROUGE-L: {metrics_summary['test_rougeL']:.4f}")
        print("="*40)

if __name__ == "__main__":
    main()
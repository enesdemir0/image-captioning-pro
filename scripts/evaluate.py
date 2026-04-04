import os
import tensorflow as tf
import numpy as np
import mlflow
import dagshub
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient # Required for the fix
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
    
    # 1. Standardized Naming
    enc_name = config['model']['encoder_name']
    dec_type = config['model']['decoder_type']
    layers = config['model']['num_layers']
    subset = config['dataset']['subset_size']
    epochs = config['training']['epochs']
    model_id = f"ENC_{enc_name}_DEC_{dec_type}_L{layers}_S{subset}_E{epochs}"
    
    # 2. Initialize DagsHub/MLflow
    dagshub.init(repo_owner=config['mlflow']['repo_owner'], repo_name=config['mlflow']['repo_name'], mlflow=True)
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])

    # --- THE FIX: RESTORE FROM TRASH IF DELETED ---
    client = MlflowClient()
    exp = client.get_experiment_by_name(model_id)
    if exp and exp.lifecycle_stage == 'deleted':
        print(f"♻️ Restoring deleted experiment: {model_id}")
        client.restore_experiment(exp.experiment_id)
    
    mlflow.set_experiment(model_id)

    # 3. Data Setup
    loader = DataLoader(config)
    img_paths, captions = loader.load_annotations()
    if subset > 0:
        img_paths, captions = img_paths[:subset], captions[:subset]

    (train_imgs, train_caps), _, (test_imgs, test_caps) = loader.split_data(img_paths, captions)
    loader.text_processor.fit_on_texts(train_caps)

    # 4. Initialize and Force Build Models
    encoder = CNN_Encoder(config)
    decoder = RNN_Decoder(config)
    encoder(tf.zeros((1, 299, 299, 3)))
    decoder(tf.zeros((1, 1)), decoder.init_decoder_state(tf.zeros((1, config['model']['units']))))

    # 5. Load Weights
    ckpt_dir = config['training']['checkpoint_path']
    enc_path = os.path.join(ckpt_dir, f"{model_id}_encoder.weights.h5")
    dec_path = os.path.join(ckpt_dir, f"{model_id}_decoder.weights.h5")

    if not os.path.exists(enc_path):
        print(f"❌ Error: Could not find weights for {model_id}")
        return

    encoder.load_weights(enc_path)
    decoder.load_weights(dec_path)
    print(f"✅ Weights loaded successfully for evaluation.")

    # 6. Evaluation Loop
    b1_l, b2_l, b3_l, b4_l, met_l, rou_l = [], [], [], [], [], []

    with mlflow.start_run(run_name="Evaluation_Phase"):
        print(f"--- Running Evaluation: {model_id} ---")
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

        # Final Summary
        summary = {
            "test_bleu1": np.mean(b1_l), "test_bleu2": np.mean(b2_l),
            "test_bleu3": np.mean(b3_l), "test_bleu4": np.mean(b4_l),
            "test_meteor": np.mean(met_l), "test_rougeL": np.mean(rou_l)
        }
        mlflow.log_metrics(summary)

        print("\n" + "="*60)
        print(f"FINAL RESULTS: {model_id}")
        print(f"BLEU-4: {summary['test_bleu4']:.4f} | METEOR: {summary['test_meteor']:.4f} | ROUGE-L: {summary['test_rougeL']:.4f}")
        print("="*60)

if __name__ == "__main__":
    main()
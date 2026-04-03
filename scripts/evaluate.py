import os
import tensorflow as tf
import numpy as np
import mlflow
import dagshub
from src.utils.config_loader import load_config
from src.data.dataset_loader import DataLoader
from src.models.encoder import CNN_Encoder
from src.models.decoder import RNN_Decoder
from src.utils.metrics import calculate_all_metrics

def generate_caption(image_tensor, encoder, decoder, text_processor, config):
    features = encoder(image_tensor)
    hidden = decoder.init_decoder_state(tf.reduce_mean(features, axis=1))
    
    # LOOKUP THE START TOKEN ID
    start_token = text_processor.tokenizer.word_index['<start>']
    dec_input = tf.expand_dims([start_token], 0)
    
    result = []
    for i in range(config['dataset']['max_caption_length']):
        preds, hidden = decoder(dec_input, hidden)
        predicted_id = tf.argmax(preds[0]).numpy()
        
        # Check if ID exists in dictionary
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
    
    # CRITICAL: Use the SAME subset logic as training to ensure the vocabulary matches!
    subset_size = config['dataset'].get('subset_size', 5000)
    img_paths, captions = img_paths[:subset_size], captions[:subset_size]

    # Split the data
    (train_imgs, train_caps), _, (test_imgs, test_caps) = loader.split_data(img_paths, captions)

    # --- FIX: BUILD THE VOCABULARY ---
    # We must 'fit' the tokenizer on the training captions so it knows the word IDs
    print("Building vocabulary from training set...")
    loader.text_processor.fit_on_texts(train_caps)

    # Initialize Models
    encoder = CNN_Encoder(config)
    decoder = RNN_Decoder(config)

    # Build and Load Weights
    encoder(tf.zeros((1, 299, 299, 3)))
    dummy_state = decoder.init_decoder_state(tf.zeros((1, config['model']['units'])))
    decoder(tf.zeros((1, 1)), dummy_state)

    ckpt_dir = config['training']['checkpoint_path']
    enc_path = os.path.join(ckpt_dir, f"{model_id}_encoder.weights.h5")
    dec_path = os.path.join(ckpt_dir, f"{model_id}_decoder.weights.h5")

    encoder.load_weights(enc_path)
    decoder.load_weights(dec_path)
    print(f"✅ Weights loaded. Vocabulary size: {len(loader.text_processor.tokenizer.word_index)}")

    bleus, meteors, rouges = [], [], []

    with mlflow.start_run(run_name=f"EVAL_{model_id}"):
        print(f"\n--- Evaluating Test Set for {model_id} ---")
        # Run on 50 samples for the report
        for i in range(min(50, len(test_imgs))):
            img, _ = loader.image_processor.preprocess_image(test_imgs[i])
            pred = generate_caption(tf.expand_dims(img, 0), encoder, decoder, loader.text_processor, config)
            
            b, m, r = calculate_all_metrics(test_caps[i], pred)
            bleus.append(b); meteors.append(m); rouges.append(r)

            if i < 5: # Visual check
                print(f"\nReal: {test_caps[i]}")
                print(f"Pred: <start> {pred} <end>")
                print(f"BLEU-4: {b:.4f}")

        avg_b, avg_m, avg_r = np.mean(bleus), np.mean(meteors), np.mean(rouges)
        mlflow.log_metrics({"test_bleu4": avg_b, "test_meteor": avg_m, "test_rougeL": avg_r})

        print(f"\n| Architecture | BLEU-4 | METEOR | ROUGE-L |")
        print(f"| {model_id} | {avg_b:.4f} | {avg_m:.4f} | {avg_r:.4f} |")

if __name__ == "__main__":
    main()
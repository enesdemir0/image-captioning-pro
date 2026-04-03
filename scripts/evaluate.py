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
    mean_features = tf.reduce_mean(features, axis=1)
    hidden = decoder.init_decoder_state(mean_features)
    
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
    
    # Init MLflow for Evaluation Logging
    dagshub.init(repo_owner=config['mlflow']['repo_owner'], repo_name=config['mlflow']['repo_name'], mlflow=True)
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    # We log evaluation as a separate run or nested in the experiment
    
    loader = DataLoader(config)
    img_paths, captions = loader.load_annotations()
    _, _, (test_imgs, test_caps) = loader.split_data(img_paths, captions)

    # Build and Load Models
    encoder = CNN_Encoder(config)
    decoder = RNN_Decoder(config)

    # --- FIX: PROPERLY BUILDING THE MODELS ---
    dummy_img = tf.zeros((1, 299, 299, 3))
    dummy_features = encoder(dummy_img)
    dummy_hidden = decoder.init_decoder_state(tf.reduce_mean(dummy_features, axis=1))
    decoder(tf.zeros((1, 1)), dummy_hidden) # Dummy call to build weights

    encoder.load_weights("models/checkpoints/encoder_e10.weights.h5")
    decoder.load_weights("models/checkpoints/decoder_e10.weights.h5")

    bleus, meteors, rouges = [], [], []

    with mlflow.start_run(run_name=f"EVAL_{config['model']['decoder_type']}"):
        print("\nEvaluating on Test Set...")
        for i in range(100): # Evaluate 100 samples
            img, _ = loader.image_processor.preprocess_image(test_imgs[i])
            pred = generate_caption(tf.expand_dims(img, 0), encoder, decoder, loader.text_processor, config)
            
            b, m, r = calculate_all_metrics(test_caps[i], pred)
            bleus.append(b); meteors.append(m); rouges.append(r)

        # Log Average Metrics to DagsHub
        avg_bleu = np.mean(bleus)
        avg_meteor = np.mean(meteors)
        avg_rouge = np.mean(rouges)
        
        mlflow.log_metric("test_bleu4", avg_bleu)
        mlflow.log_metric("test_meteor", avg_meteor)
        mlflow.log_metric("test_rougeL", avg_rouge)

        print(f"\nFinal Average Metrics:")
        print(f"BLEU-4: {avg_bleu:.4f} | METEOR: {avg_meteor:.4f} | ROUGE-L: {avg_rouge:.4f}")

if __name__ == "__main__":
    main()
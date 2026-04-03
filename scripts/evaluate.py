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
    dec_input = tf.expand_dims([text_processor.tokenizer.word_index['<start>']], 0)
    
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
    _, _, (test_imgs, test_caps) = loader.split_data(img_paths, captions)

    # Initialize and Load
    encoder, decoder = CNN_Encoder(config), RNN_Decoder(config)
    encoder(tf.zeros((1, 299, 299, 3)))
    decoder.init_decoder_state(tf.zeros((1, config['model']['units'])))
    
    ckpt_path = os.path.join(config['training']['checkpoint_path'], f"{model_id}_encoder.weights.h5")
    if not os.path.exists(ckpt_path):
        print(f"❌ Error: Weights not found for {model_id} at {ckpt_path}")
        return

    encoder.load_weights(ckpt_path)
    decoder.load_weights(os.path.join(config['training']['checkpoint_path'], f"{model_id}_decoder.weights.h5"))

    bleus, meteors, rouges = [], [], []

    with mlflow.start_run(run_name=f"EVAL_{model_id}"):
        print(f"\n--- Evaluating Test Set for {model_id} ---")
        # Run on first 100 images of the Test Set
        for i in range(100):
            img, _ = loader.image_processor.preprocess_image(test_imgs[i])
            pred = generate_caption(tf.expand_dims(img, 0), encoder, decoder, loader.text_processor, config)
            
            b, m, r = calculate_all_metrics(test_caps[i], pred)
            bleus.append(b); meteors.append(m); rouges.append(r)

        avg_b, avg_m, avg_r = np.mean(bleus), np.mean(meteors), np.mean(rouges)
        mlflow.log_metrics({"test_bleu4": avg_b, "test_meteor": avg_m, "test_rougeL": avg_r})

        print(f"\n| Architecture | BLEU-4 | METEOR | ROUGE-L |")
        print(f"| {model_id} | {avg_b:.4f} | {avg_m:.4f} | {avg_r:.4f} |")

if __name__ == "__main__":
    main()
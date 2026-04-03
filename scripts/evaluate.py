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
    dagshub.init(repo_owner=config['mlflow']['repo_owner'], repo_name=config['mlflow']['repo_name'], mlflow=True)
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    
    loader = DataLoader(config)
    img_paths, captions = loader.load_annotations()
    _, _, (test_imgs, test_caps) = loader.split_data(img_paths, captions)

    # Initialize and Build Models
    encoder, decoder = CNN_Encoder(config), RNN_Decoder(config)
    encoder(tf.zeros((1, 299, 299, 3)))
    decoder.init_decoder_state(tf.zeros((1, config['model']['units'])))
    
    # Load Weights
    encoder.load_weights("models/checkpoints/encoder_e10.weights.h5")
    decoder.load_weights("models/checkpoints/decoder_e10.weights.h5")

    bleus, meteors, rouges = [], [], []

    with mlflow.start_run(run_name=f"FINAL_REPORT_{config['model']['decoder_type']}"):
        print("\nEvaluating on Test Set...")
        os.makedirs("results/samples", exist_ok=True)

        for i in range(50): # Evaluate 50 samples
            img, _ = loader.image_processor.preprocess_image(test_imgs[i])
            pred = generate_caption(tf.expand_dims(img, 0), encoder, decoder, loader.text_processor, config)
            
            b, m, r = calculate_all_metrics(test_caps[i], pred)
            bleus.append(b); meteors.append(m); rouges.append(r)

            # SAVE FIGURES FOR THE REPORT (Like Figure 5/6)
            if i < 10: # Save first 10 as examples
                plt.figure(figsize=(8,8))
                plt.imshow(img.numpy() * 0.5 + 0.5) # De-normalize for viewing
                plt.title(f"Real: {test_caps[i]}\nPred: {pred}\nBLEU: {b:.2f}")
                plt.axis('off')
                filename = f"results/samples/sample_{i}.png"
                plt.savefig(filename)
                plt.close()
                mlflow.log_artifact(filename) # Log to DagsHub

        # Log Metrics
        mlflow.log_metric("test_bleu4", np.mean(bleus))
        mlflow.log_metric("test_meteor", np.mean(meteors))
        mlflow.log_metric("test_rougeL", np.mean(rouges))

        print(f"\nFinal Averages -> BLEU-4: {np.mean(bleus):.4f} | METEOR: {np.mean(meteors):.4f} | ROUGE-L: {np.mean(rouges):.4f}")

if __name__ == "__main__":
    main()
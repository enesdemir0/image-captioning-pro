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

def generate_caption_smart(image_tensor, encoder, decoder, text_processor, config):
    features = encoder(image_tensor)
    hidden = decoder.init_decoder_state(tf.reduce_mean(features, axis=1))
    
    start_token = text_processor.tokenizer.word_index['<start>']
    dec_input = tf.expand_dims([start_token], 0)
    
    result = []
    attention_plot = np.zeros((config['dataset']['max_caption_length'], features.shape[1]))
    
    # REPETITION PENALTY LOGIC
    used_word_ids = []

    for i in range(config['dataset']['max_caption_length']):
        preds, hidden, attn_weights = decoder(dec_input, features, hidden)
        
        if attn_weights is not None:
            attention_plot[i] = tf.reshape(attn_weights, (-1,)).numpy()

        # Get logits
        logits = preds[0].numpy()
        
        # APPLY PENALTY: If we already said a word, make it very unlikely to say it again
        for word_id in used_word_ids:
            logits[word_id] -= 2.0 
        
        predicted_id = np.argmax(logits)
        word = text_processor.tokenizer.index_word.get(predicted_id, '<unk>')
        
        if word == '<end>': break
        
        result.append(word)
        used_word_ids.append(predicted_id) # Remember this word
        dec_input = tf.expand_dims([predicted_id], 0)
        
    return ' '.join(result), attention_plot

def main():
    config = load_config()
    # Ensure we look for the 512 unit models!
    enc, dec, layers, units = config['model']['encoder_name'], config['model']['decoder_type'], config['model']['num_layers'], 512
    subset, epochs = config['dataset']['subset_size'], 50
    attn = config['model'].get('attention_type', 'scaled_dot')
    model_id = f"ENC_{enc}_DEC_{dec}_L{layers}_S{subset}_E{epochs}_TF_{attn}"
    
    dagshub.init(repo_owner=config['mlflow']['repo_owner'], repo_name=config['mlflow']['repo_name'], mlflow=True)
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(model_id)

    loader = DataLoader(config)
    img_paths, captions = loader.load_annotations()
    (train_imgs, _), _, (test_imgs, test_caps) = loader.split_data(img_paths[:subset], captions[:subset])
    loader.text_processor.fit_on_texts(train_imgs)

    encoder, decoder = CNN_Encoder(config), RNN_Decoder(config)
    # Build
    encoder(tf.zeros((1, 299, 299, 3)))
    num_f = 100 if enc == "Xception" else 64
    decoder(tf.zeros((1, 1)), tf.zeros((1, num_f, 512)), decoder.init_decoder_state(tf.zeros((1, 512))))

    # Load the 512 unit weights
    ckpt_dir = config['training']['checkpoint_path']
    encoder.load_weights(os.path.join(ckpt_dir, f"{model_id}_encoder.weights.h5"))
    decoder.load_weights(os.path.join(ckpt_dir, f"{model_id}_decoder.weights.h5"))

    print(f"✅ Evaluating the BEST model: {model_id}")
    b1_l, b2_l, b3_l, b4_l, met_l, rou_l = [], [], [], [], [], []

    with mlflow.start_run(run_name="Final_Clean_Evaluation"):
        os.makedirs("results/samples", exist_ok=True)
        for i in range(min(100, len(test_imgs))):
            img_tensor, _ = loader.image_processor.preprocess_image(test_imgs[i])
            pred, attn_plot = generate_caption_smart(tf.expand_dims(img_tensor, 0), encoder, decoder, loader.text_processor, config)
            
            (b1, b2, b3, b4), m, r = calculate_all_metrics(test_caps[i], pred)
            b1_l.append(b1); b2_l.append(b2); b3_l.append(b3); b4_l.append(b4); met_l.append(m); rou_l.append(r)

            if i < 5:
                # Save visual sample
                plt.figure(figsize=(10, 10))
                plt.imshow(img_tensor.numpy() * 0.5 + 0.5)
                plt.title(f"REAL: {test_caps[i][7:-5]}\nPRED: {pred}\nBLEU-4: {b4:.4f}")
                plt.savefig(f"results/samples/Sample_{i}.png"); plt.close()
                mlflow.log_artifact(f"results/samples/Sample_{i}.png")

        mlflow.log_metrics({"test_bleu4": np.mean(b4_l), "test_meteor": np.mean(met_l), "test_rougeL": np.mean(rou_l)})
        print(f"Final Success! BLEU-4: {np.mean(b4_l):.4f} | METEOR: {np.mean(met_l):.4f}")

if __name__ == "__main__":
    main()
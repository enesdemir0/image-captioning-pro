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
    
    # MATCH THE AUTOMATIC NAME
    enc_name = config['model']['encoder_name']
    dec_type = config['model']['decoder_type']
    layers = config['model']['num_layers']
    model_id = f"Encoder_{enc_name}_Decoder_{dec_type}_L{layers}"
    
    dagshub.init(repo_owner=config['mlflow']['repo_owner'], repo_name=config['mlflow']['repo_name'], mlflow=True)
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    
    # MATCH THE EXPERIMENT
    mlflow.set_experiment(model_id)

    loader = DataLoader(config)
    img_paths, captions = loader.load_annotations()
    
    subset_size = config['dataset'].get('subset_size', 0)
    if subset_size > 0:
        img_paths, captions = img_paths[:subset_size], captions[:subset_size]

    (tr_imgs, tr_caps), _, (test_imgs, test_caps) = loader.split_data(img_paths, captions)
    loader.text_processor.fit_on_texts(tr_caps)

    encoder, decoder = CNN_Encoder(config), RNN_Decoder(config)
    encoder(tf.zeros((1, 299, 299, 3)))
    decoder(tf.zeros((1, 1)), decoder.init_decoder_state(tf.zeros((1, config['model']['units']))))

    ckpt_dir = config['training']['checkpoint_path']
    encoder.load_weights(os.path.join(ckpt_dir, f"{model_id}_encoder.weights.h5"))
    decoder.load_weights(os.path.join(ckpt_dir, f"{model_id}_decoder.weights.h5"))

    bleus, meteors, rouges = [], [], []

    # Run name "Evaluation" inside that architecture's experiment
    with mlflow.start_run(run_name="Evaluation_Phase"):
        print(f"--- EVALUATING: {model_id} ---")
        os.makedirs("results/samples", exist_ok=True)

        for i in range(min(50, len(test_imgs))):
            img_tensor, _ = loader.image_processor.preprocess_image(test_imgs[i])
            pred = generate_caption(tf.expand_dims(img_tensor, 0), encoder, decoder, loader.text_processor, config)
            b, m, r = calculate_all_metrics(test_caps[i], pred)
            bleus.append(b); meteors.append(m); rouges.append(r)

            if i < 5:
                plt.figure(figsize=(10, 8))
                plt.imshow(img_tensor.numpy() * 0.5 + 0.5)
                c_real = test_caps[i].replace('<start>', '').replace('<end>', '').strip()
                c_pred = pred.replace('<start>', '').replace('<end>', '').strip()
                plt.title(f"REAL: {c_real}\nPRED: {c_pred}\nBLEU-4: {b:.4f}")
                plt.axis('off')
                
                fig_name = f"Sample_{i}.png"
                plt.savefig(f"results/samples/{fig_name}")
                plt.close()
                mlflow.log_artifact(f"results/samples/{fig_name}")

        avg_b, avg_m, avg_r = np.mean(bleus), np.mean(meteors), np.mean(rouges)
        mlflow.log_metrics({"test_bleu4": avg_b, "test_meteor": avg_m, "test_rougeL": avg_r})
        print(f"DONE | BLEU-4: {avg_b:.4f}")

if __name__ == "__main__":
    main()
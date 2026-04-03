import tensorflow as tf
import numpy as np
import nltk
from src.utils.config_loader import load_config
from src.data.dataset_loader import DataLoader
from src.models.encoder import CNN_Encoder
from src.models.decoder import RNN_Decoder
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Download necessary resources for metrics
nltk.download('wordnet')

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
    loader = DataLoader(config)
    img_paths, captions = loader.load_annotations()
    _, _, (test_imgs, test_caps) = loader.split_data(img_paths, captions)

    # Initialize and Load Best Weights
    encoder, decoder = CNN_Encoder(config), RNN_Decoder(config)
    # Build models with a dummy input first
    encoder(tf.zeros((1, 299, 299, 3)))
    decoder.init_decoder_state(tf.zeros((1, config['model']['units'])))
    
    encoder.load_weights("models/checkpoints/encoder_e10.weights.h5")
    decoder.load_weights("models/checkpoints/decoder_e10.weights.h5")

    bleu_scores = []
    print("\n| Sample Image | Real Caption | Predicted Caption | BLEU-4 |")
    print("|--------------|--------------|-------------------|--------|")

    for i in range(20): # Test on 20 samples for the report
        img, _ = loader.image_processor.preprocess_image(test_imgs[i])
        pred = generate_caption(tf.expand_dims(img, 0), encoder, decoder, loader.text_processor, config)
        
        # Calculate BLEU-4
        score = sentence_bleu([test_caps[i].split()], pred.split(), weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)
        bleu_scores.append(score)
        
        if i < 5: # Show first 5 in the terminal
            print(f"| {i} | {test_caps[i]} | {pred} | {score:.4f} |")

    print(f"\nFinal Average BLEU-4 on Test Set: {np.mean(bleu_scores):.4f}")

if __name__ == "__main__":
    main()
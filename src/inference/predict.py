import numpy as np
import tensorflow as tf
from PIL import Image


_PREPROCESS = {
    "Xception":    (tf.keras.applications.xception.preprocess_input,    (299, 299)),
    "InceptionV3": (tf.keras.applications.inception_v3.preprocess_input,(299, 299)),
    "VGG16":       (tf.keras.applications.vgg16.preprocess_input,       (224, 224)),
    "ResNet50":    (tf.keras.applications.resnet50.preprocess_input,    (224, 224)),
}


class CNNRNNPredictor:
    """Greedy-decode inference wrapper for the custom CNN+RNN captioning model.

    Accepts a PIL Image directly (no file-path needed), applies backbone-specific
    preprocessing, and runs the encoder→decoder pipeline to produce a caption string.
    """

    def __init__(self, encoder, decoder, text_processor, config):
        self.encoder        = encoder
        self.decoder        = decoder
        self.text_processor = text_processor
        self.config         = config

        enc_name = config['model']['encoder_name']
        self._preprocess_fn, self._target_size = _PREPROCESS.get(
            enc_name, _PREPROCESS["Xception"]
        )

    def predict(self, image_pil: Image.Image, max_length: int = None) -> str:
        if max_length is None:
            max_length = int(self.config['dataset']['max_caption_length'])

        img    = image_pil.resize(self._target_size).convert("RGB")
        arr    = np.array(img, dtype=np.float32)
        tensor = self._preprocess_fn(tf.expand_dims(arr, 0))   # (1, H, W, 3)

        features = self.encoder(tensor)
        hidden   = self.decoder.init_decoder_state(features)

        tok       = self.text_processor.tokenizer
        start_idx = tok.word_index.get('<start>', 1)
        end_idx   = tok.word_index.get('<end>',   2)
        dec_input = tf.expand_dims([start_idx], 0)

        words = []
        for _ in range(max_length):
            predictions, hidden, _ = self.decoder(dec_input, features, hidden)
            pred_id = int(tf.argmax(predictions[0]).numpy())
            if pred_id == end_idx:
                break
            word = tok.index_word.get(pred_id, '')
            if word and word not in ('<start>', '<end>', '<unk>', '<pad>'):
                words.append(word)
            dec_input = tf.expand_dims([pred_id], 0)

        return ' '.join(words)

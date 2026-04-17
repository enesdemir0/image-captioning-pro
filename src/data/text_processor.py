import re
import json
import tensorflow as tf


class TextProcessor:
    def __init__(self, config):
        self.vocab_size = config['dataset']['vocab_size']
        self.max_length = int(config['dataset']['max_caption_length'])
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=self.vocab_size,
            filters='!"#$%&()*+.,-/:;=?@[\\]^_`{|}~ ',
            lower=True,
            oov_token="<unk>"
        )

    def clean_caption(self, caption):
        caption = str(caption).lower()
        caption = re.sub(r'[^\w\s]', '', caption)
        caption = f"<start> {caption} <end>"
        return re.sub(r'\s+', ' ', caption).strip()

    def fit_on_texts(self, captions):
        if not captions:
            return False
        self.tokenizer.fit_on_texts(captions)
        self.tokenizer.word_index['<pad>'] = 0
        self.tokenizer.index_word[0] = '<pad>'
        return True

    def tokenize_and_pad(self, captions):
        captions = [str(c) for c in captions]
        sequences = self.tokenizer.texts_to_sequences(captions)
        return tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=self.max_length, padding='post', dtype='int32'
        )

    def save_tokenizer(self, path):
        """Serializes vocabulary to a portable plain-JSON file.
        Uses explicit word_index/index_word dicts — immune to Keras version drift."""
        data = {
            'word_index': self.tokenizer.word_index,
            'index_word': {str(k): v for k, v in self.tokenizer.index_word.items()},
            'vocab_size': self.vocab_size,
            'max_length': self.max_length
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Tokenizer saved → {path}")

    def load_tokenizer(self, path):
        """Restores exact word↔index mapping from JSON — no re-fitting, no index drift."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.tokenizer.word_index = data['word_index']
        self.tokenizer.index_word = {int(k): v for k, v in data['index_word'].items()}
        print(f"Tokenizer loaded ← {path}")

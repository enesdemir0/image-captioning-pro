import re
import tensorflow as tf
import numpy as np

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
        caption = re.sub(r'\s+', ' ', caption).strip()
        return caption

    def fit_on_texts(self, captions):
        """Builds the word index. Returns True if successful."""
        if not captions:
            return False
        self.tokenizer.fit_on_texts(captions)
        self.tokenizer.word_index['<pad>'] = 0
        self.tokenizer.index_word[0] = '<pad>'
        return True

    def tokenize_and_pad(self, captions):
        """Converts text to padded sequences safely."""
        # Ensure all captions are strings
        captions = [str(c) for c in captions]
        sequences = self.tokenizer.texts_to_sequences(captions)
        
        # If tokenizer isn't fitted, sequences will be empty lists.
        # We ensure it returns a valid numpy array of zeros in that case.
        return tf.keras.preprocessing.sequence.pad_sequences(
            sequences, 
            maxlen=self.max_length, 
            padding='post', 
            dtype='int32'
        )
import re
import tensorflow as tf

class TextProcessor:
    """
    Handles cleaning, tokenizing, and padding of captions.
    Optimized for GPU masking and professional NLP workflows.
    """
    def __init__(self, config):
        self.vocab_size = config['dataset']['vocab_size']
        self.max_length = config['dataset']['max_caption_length']
        
        # We exclude < and > from filters so our <start> and <end> tags stay safe!
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=self.vocab_size,
            filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ', 
            lower=True,
            oov_token="<unk>"
        )

    def clean_caption(self, caption):
        """Standardizes text: lowercase, removes punctuation, adds boundary tokens."""
        caption = caption.lower()
        # Remove punctuation except for spaces
        caption = re.sub(r'[^\w\s]', '', caption)
        # Add boundary tokens AFTER cleaning to ensure they stay intact
        caption = f"<start> {caption} <end>"
        # Remove extra whitespace
        caption = re.sub(r'\s+', ' ', caption).strip()
        return caption

    def fit_on_texts(self, captions):
        """Creates the word-to-index dictionary."""
        self.tokenizer.fit_on_texts(captions)
        # Keras reserves 0 for padding automatically. 
        # We don't need to force it, but we ensure our code knows 0 = <pad>
        self.tokenizer.word_index['<pad>'] = 0
        self.tokenizer.index_word[0] = '<pad>'

    def tokenize_and_pad(self, captions):
        """Converts text captions to padded integer sequences for the Decoder."""
        sequences = self.tokenizer.texts_to_sequences(captions)
        # 'post' padding is industry standard for Encoder-Decoder Attention models
        return tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=self.max_length, padding='post'
        )
import re
import tensorflow as tf

class TextProcessor:
    """
    Handles cleaning, tokenizing, and padding of captions.
    """
    def __init__(self, config):
        # Read values from config
        self.vocab_size = config['dataset']['vocab_size']
        self.max_length = config['dataset']['max_caption_length']
        
        # Initialize the Keras Tokenizer
        # In src/data/text_processor.py
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=self.vocab_size,
            filters=r'!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ', # Added 'r' here
            lower=True,
            oov_token="<unk>"
        )
        

    def clean_caption(self, caption):
        """Standardizes the text: lowercase, removes punctuation, adds tokens."""
        caption = caption.lower()
        caption = re.sub(r'[^\w\s]', '', caption) # Remove punctuation
        caption = f"<start> {caption} <end>"     # Add boundary tokens
        caption = re.sub(r'\s+', ' ', caption).strip() # Remove extra spaces
        return caption

    def fit_on_texts(self, captions):
        """Creates the word-to-index dictionary."""
        self.tokenizer.fit_on_texts(captions)
        # Manually ensure padding token is at index 0
        self.tokenizer.word_index['<pad>'] = 0
        self.tokenizer.index_word[0] = '<pad>'

    def tokenize_and_pad(self, captions):
        """Converts text captions to padded integer sequences."""
        sequences = self.tokenizer.texts_to_sequences(captions)
        return tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=self.max_length, padding='post'
        )
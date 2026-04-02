import re
import tensorflow as tf

class TextPreprocessor:
    """
    Handles cleaning, tokenizing, and padding of captions.
    """
    def __init__(self, vocab_size=5000, max_length=20):
        self.vocab_size = vocab_size
        self.max_length = max_length
        # The tokenizer will turn words into numbers
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            num_words=vocab_size,
            filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ',
            lower=True,
            oov_token="<unk>" # Out Of Vocabulary token
        )

    def clean_caption(self, caption):
        """Removes special characters and adds start/end tokens."""
        caption = caption.lower()
        # Remove punctuation using Regex
        caption = re.sub(r'[^\w\s]', '', caption)
        # Add start and end tokens so the model knows the sequence boundaries
        caption = f"<start> {caption} <end>"
        # Remove extra spaces
        caption = re.sub(r'\s+', ' ', caption).strip()
        return caption

    def fit_on_texts(self, captions):
        """Creates the word-to-index mapping based on provided captions."""
        self.tokenizer.fit_on_texts(captions)
        # Ensure '<pad>' is at index 0 (standard practice)
        self.tokenizer.word_index['<pad>'] = 0
        self.tokenizer.index_word[0] = '<pad>'

    def tokenize_and_pad(self, captions):
        """Converts text captions to sequences of padded integers."""
        sequences = self.tokenizer.texts_to_sequences(captions)
        padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
            sequences, maxlen=self.max_length, padding='post'
        )
        return padded_sequences
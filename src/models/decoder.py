import tensorflow as tf

class RNN_Decoder(tf.keras.Model):
    """
    A Stacked RNN Decoder (LSTM or GRU) that initializes hidden states 
    using the Encoder's features.
    """
    def __init__(self, config):
        super(RNN_Decoder, self).__init__()
        self.units = config['model']['units']
        self.vocab_size = config['dataset']['vocab_size']
        self.embedding_dim = config['model']['embedding_dim']
        self.cell_type = config['model']['decoder_type'].upper() # Force uppercase
        self.num_layers = config['model']['num_layers']
        
        # 1. Word Embedding Layer
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)

        # 2. Build the Stacked RNN Cells
        if self.cell_type == "LSTM":
            cells = [tf.keras.layers.LSTMCell(self.units) for _ in range(self.num_layers)]
        else: # GRU
            cells = [tf.keras.layers.GRUCell(self.units) for _ in range(self.num_layers)]
        
        self.stacked_rnn = tf.keras.layers.StackedRNNCells(cells)
        self.rnn = tf.keras.layers.RNN(self.stacked_rnn, return_sequences=True, return_state=True)

        # 3. Output Projection
        self.fc1 = tf.keras.layers.Dense(self.units, activation='relu')
        self.fc2 = tf.keras.layers.Dense(self.vocab_size)

    def call(self, x, hidden_state):
        # Embed the word
        x = self.embedding(x)

        # Pass through the RNN stack
        # next_state will be a list of tensors
        output, *next_state = self.rnn(x, initial_state=hidden_state)

        # Prediction
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2])) 
        logits = self.fc2(x)

        return logits, next_state

    def init_decoder_state(self, encoder_output):
        """
        Initializes the state for every layer.
        LSTM needs 2 tensors per layer ([h, c, h, c...])
        GRU needs 1 tensor per layer ([h, h, h...])
        """
        if self.cell_type == "LSTM":
            # For 3 layers, we need a list of 6 tensors
            return [encoder_output for _ in range(self.num_layers * 2)]
        else:
            # For 3 layers, we need a list of 3 tensors
            return [encoder_output for _ in range(self.num_layers)]
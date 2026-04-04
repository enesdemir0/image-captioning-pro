import tensorflow as tf

class RNN_Decoder(tf.keras.Model):
    """
    A Universal Stacked RNN Decoder.
    Supports any number of layers and automatically handles the 
    different state structures of LSTM and GRU.
    """
    def __init__(self, config):
        super(RNN_Decoder, self).__init__()
        self.units = config['model']['units']
        self.vocab_size = config['dataset']['vocab_size']
        self.embedding_dim = config['model']['embedding_dim']
        self.cell_type = config['model']['decoder_type'].upper()
        self.num_layers = config['model']['num_layers']
        
        # 1. Word Embedding
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)

        # 2. Build the Cell Stack
        if self.cell_type == "LSTM":
            cells = [tf.keras.layers.LSTMCell(self.units) for _ in range(self.num_layers)]
        else: # GRU
            cells = [tf.keras.layers.GRUCell(self.units) for _ in range(self.num_layers)]
        
        self.stacked_rnn_cells = tf.keras.layers.StackedRNNCells(cells)
        
        # We use the RNN layer to wrap our stack
        # return_state=True is what allows us to pass memory word-by-word
        self.rnn_layer = tf.keras.layers.RNN(
            self.stacked_rnn_cells, 
            return_sequences=True, 
            return_state=True
        )

        # 3. Output Projection
        self.fc1 = tf.keras.layers.Dense(self.units, activation='relu')
        self.fc2 = tf.keras.layers.Dense(self.vocab_size)

    def call(self, x, hidden_state):
        # x shape: (batch_size, 1)
        x = self.embedding(x)

        # THE UNIVERSAL FIX: 
        # In a Stacked RNN, Keras returns [Output, State_L1, State_L2, ... State_Ln]
        # We catch everything in a single list called 'rnn_results'
        rnn_results = self.rnn_layer(x, initial_state=hidden_state)

        # The first item is always the prediction output
        output = rnn_results[0]
        
        # Everything from index 1 onwards are the states for the next step
        # This works for 1 layer, 3 layers, or 100 layers!
        next_state = rnn_results[1:]

        # Project to vocabulary
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2])) 
        logits = self.fc2(x)

        return logits, next_state

    def init_decoder_state(self, encoder_output):
        """
        Creates the exact nested structure Keras expects for the stack.
        """
        if self.cell_type == "LSTM":
            # LSTM expects a list of lists: [[h, c], [h, c], [h, c]]
            return [[encoder_output, encoder_output] for _ in range(self.num_layers)]
        else:
            # GRU expects a flat list: [h, h, h]
            return [encoder_output for _ in range(self.num_layers)]
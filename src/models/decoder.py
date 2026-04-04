import tensorflow as tf

class RNN_Decoder(tf.keras.Model):
    """
    A robust Stacked RNN Decoder that handles variable layers and 
    nested states for both LSTM and GRU in Keras 3.
    """
    def __init__(self, config):
        super(RNN_Decoder, self).__init__()
        self.units = config['model']['units']
        self.vocab_size = config['dataset']['vocab_size']
        self.embedding_dim = config['model']['embedding_dim']
        self.cell_type = config['model']['decoder_type'].upper()
        self.num_layers = config['model']['num_layers']
        
        # 1. Word Embedding Layer
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)

        # 2. Build the Stacked RNN Cells
        if self.cell_type == "LSTM":
            cells = [tf.keras.layers.LSTMCell(self.units) for _ in range(self.num_layers)]
        else: # GRU
            cells = [tf.keras.layers.GRUCell(self.units) for _ in range(self.num_layers)]
        
        self.stacked_rnn = tf.keras.layers.StackedRNNCells(cells)
        # return_state=True is required to pass memory word-by-word
        self.rnn = tf.keras.layers.RNN(self.stacked_rnn, return_sequences=True, return_state=True)

        # 3. Output Projection
        self.fc1 = tf.keras.layers.Dense(self.units, activation='relu')
        self.fc2 = tf.keras.layers.Dense(self.vocab_size)

    def call(self, x, hidden_state):
        # Embed the word -> (batch, 1, embedding_dim)
        x = self.embedding(x)

        # THE FIX: Run the RNN and catch everything in one variable
        # For a 3-layer model, rnn_output will contain [Output, State_L1, State_L2, State_L3]
        rnn_results = self.rnn(x, initial_state=hidden_state)
        
        # The first element is always the word prediction output
        output = rnn_results[0]
        # Everything else is the state data for the next word
        next_state = rnn_results[1:] 

        # Final projection to vocabulary
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2])) 
        logits = self.fc2(x)

        return logits, next_state

    def init_decoder_state(self, encoder_output):
        """
        Initializes the nested state structure required by StackedRNNCells.
        """
        if self.cell_type == "LSTM":
            # LSTM needs [ [h,c], [h,c], [h,c] ] for 3 layers
            return [[encoder_output, encoder_output] for _ in range(self.num_layers)]
        else:
            # GRU needs [ h, h, h ] for 3 layers
            return [encoder_output for _ in range(self.num_layers)]
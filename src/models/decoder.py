import tensorflow as tf

class RNN_Decoder(tf.keras.Model):
    """
    A Stacked RNN Decoder that correctly handles nested states for LSTM/GRU.
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
        # return_state=True returns the nested state structure
        self.rnn = tf.keras.layers.RNN(self.stacked_rnn, return_sequences=True, return_state=True)

        # 3. Output Projection
        self.fc1 = tf.keras.layers.Dense(self.units, activation='relu')
        self.fc2 = tf.keras.layers.Dense(self.vocab_size)

    def call(self, x, hidden_state):
        # x shape: (batch_size, 1)
        x = self.embedding(x)

        # In Keras 3 Stacked RNN, it returns (output, final_states)
        # where final_states matches the structure of initial_state
        output, next_state = self.rnn(x, initial_state=hidden_state)

        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2])) 
        logits = self.fc2(x)

        return logits, next_state

    def init_decoder_state(self, encoder_output):
        """
        Creates the correct nested state structure.
        """
        if self.cell_type == "LSTM":
            # MUST be a list of lists: [[h, c], [h, c], [h, c]]
            return [[encoder_output, encoder_output] for _ in range(self.num_layers)]
        else:
            # Flat list for GRU: [h, h, h]
            return [encoder_output for _ in range(self.num_layers)]
import tensorflow as tf

class RNN_Decoder(tf.keras.Model):
    """
    A Stacked RNN Decoder (LSTM or GRU) that initializes hidden states 
    using the Encoder's features (the 'Dense Map').
    """
    def __init__(self, config):
        super(RNN_Decoder, self).__init__()
        self.units = config['model']['units']
        self.vocab_size = config['dataset']['vocab_size']
        self.embedding_dim = config['model']['embedding_dim']
        self.cell_type = config['model']['decoder_type']
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

        # 3. Output Projection (The Dense layers for prediction)
        self.fc1 = tf.keras.layers.Dense(self.units, activation='relu')
        self.fc2 = tf.keras.layers.Dense(self.vocab_size)

    def call(self, x, hidden_state):
        """
        x: Current word token (batch_size, 1)
        hidden_state: List of states for each layer in the stack
        """
        # Step 1: Embed the word -> (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # Step 2: Pass through the N-layer stack
        # output shape: (batch_size, 1, units)
        output, *next_state = self.rnn(x, initial_state=hidden_state)

        # Step 3: Prediction (Logits)
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2])) # Flatten for Dense
        logits = self.fc2(x)

        return logits, next_state

    def init_decoder_state(self, encoder_output):
        """
        Initializes the state of EVERY layer in the stack using the 
        Encoder's output (Dense Map).
        """
        # If it's an LSTM, every layer needs 2 tensors (h and c)
        if self.cell_type == "LSTM":
            state = []
            for _ in range(self.num_layers):
                state.append(encoder_output) # Hidden state (h)
                state.append(encoder_output) # Cell state (c)
            return state
        
        # If it's a GRU, every layer needs 1 tensor
        else:
            return [encoder_output for _ in range(self.num_layers)]
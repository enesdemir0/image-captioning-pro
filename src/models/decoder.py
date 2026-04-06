import tensorflow as tf
from src.models.attention import GlobalAttention # Import our new tool

class RNN_Decoder(tf.keras.Model):
    def __init__(self, config):
        super(RNN_Decoder, self).__init__()
        self.units = config['model']['units']
        self.vocab_size = config['dataset']['vocab_size']
        self.embedding_dim = config['model']['embedding_dim']
        self.cell_type = config['model']['decoder_type'].upper()
        self.num_layers = config['model']['num_layers']
        
        # --- NEW: Check if we are using Attention ---
        self.attn_type = config['model'].get('attention_type', 'None')
        if self.attn_type != 'None':
            self.attention = GlobalAttention(config)

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)

        if self.cell_type == "LSTM":
            cells = [tf.keras.layers.LSTMCell(self.units) for _ in range(self.num_layers)]
        else:
            cells = [tf.keras.layers.GRUCell(self.units) for _ in range(self.num_layers)]
        
        self.stacked_rnn_cells = tf.keras.layers.StackedRNNCells(cells)
        self.rnn_layer = tf.keras.layers.RNN(self.stacked_rnn_cells, return_sequences=True, return_state=True)

        self.fc1 = tf.keras.layers.Dense(self.units, activation='relu')
        self.fc2 = tf.keras.layers.Dense(self.vocab_size)

    def call(self, x, features, hidden_state):
        """
        features: The grid of image features (batch, 64, units)
        hidden_state: The current memory
        """
        # 1. If using Attention, calculate the 'Context' (Where to look)
        if self.attn_type != 'None':
            # For stacked RNNs, we usually use the state of the LAST layer to calculate attention
            # LSTM state is [h, c], GRU is just [h]
            last_layer_state = hidden_state[-1][0] if self.cell_type == "LSTM" else hidden_state[-1]
            context_vector, attention_weights = self.attention(features, last_layer_state)
        else:
            # Phase 1 & 2: Just use the average of the features
            context_vector = tf.reduce_mean(features, axis=1)
            attention_weights = None

        # 2. Embed the word
        x = self.embedding(x)

        # 3. Combine Word + Image Context
        # x shape after concat: (batch, 1, embedding_dim + units)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # 4. Pass through RNN
        rnn_results = self.rnn_layer(x, initial_state=hidden_state)
        output = rnn_results[0]
        next_state = rnn_results[1:]

        # 5. Predict
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2])) 
        logits = self.fc2(x)

        return logits, next_state, attention_weights

    def init_decoder_state(self, encoder_output):
        if self.cell_type == "LSTM":
            return [[encoder_output, encoder_output] for _ in range(self.num_layers)]
        else:
            return [encoder_output for _ in range(self.num_layers)]
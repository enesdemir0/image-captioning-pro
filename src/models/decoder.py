import tensorflow as tf
from src.models.attention import GlobalAttention, RegionAttention

class RNN_Decoder(tf.keras.Model):
    def __init__(self, config):
        super(RNN_Decoder, self).__init__()
        self.units = config['model']['units']
        self.vocab_size = config['dataset']['vocab_size']
        self.embedding_dim = config['model']['embedding_dim']
        self.cell_type = config['model']['decoder_type'].upper()
        self.num_layers = config['model']['num_layers']
        
        # --- PHASE 3 & BONUS 2 SELECTION ---
        self.attn_type = config['model'].get('attention_type', 'None').lower()
        
        if self.attn_type == 'ran':
            # Bonus #2: Region Attention
            self.attention = RegionAttention(config)
        elif self.attn_type in ['bahdanau', 'dot', 'scaled_dot']:
            # Phase 3: Global Attention
            self.attention = GlobalAttention(config)
        else:
            self.attention = None

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
        # 1. Calculate Attention Context
        if self.attention is not None:
            # We use the hidden state of the LAST layer for attention calculation
            last_layer_state = hidden_state[-1][0] if self.cell_type == "LSTM" else hidden_state[-1]
            context_vector, attention_weights = self.attention(features, last_layer_state)
        else:
            context_vector = tf.reduce_mean(features, axis=1)
            attention_weights = None

        # 2. Embedding and Concat
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # 3. RNN Forward Pass
        rnn_results = self.rnn_layer(x, initial_state=hidden_state)
        output = rnn_results[0]
        next_state = rnn_results[1:]

        # 4. Final Prediction
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2])) 
        logits = self.fc2(x)

        return logits, next_state, attention_weights

    def init_decoder_state(self, encoder_output):
        if self.cell_type == "LSTM":
            return [[encoder_output, encoder_output] for _ in range(self.num_layers)]
        else:
            return [encoder_output for _ in range(self.num_layers)]
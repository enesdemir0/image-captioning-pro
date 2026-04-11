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
        self.attn_type = config['model'].get('attention_type', 'None').lower()
        
        # Select Attention Mechanism
        if self.attn_type == 'ran':
            self.attention = RegionAttention(config)
        elif self.attn_type in ['bahdanau', 'dot', 'scaled_dot']:
            self.attention = GlobalAttention(config)
        else:
            self.attention = None

        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)

        # Build Multi-layer RNN
        if self.cell_type == "LSTM":
            cells = [tf.keras.layers.LSTMCell(self.units) for _ in range(self.num_layers)]
        else:
            cells = [tf.keras.layers.GRUCell(self.units) for _ in range(self.num_layers)]
        
        self.stacked_rnn_cells = tf.keras.layers.StackedRNNCells(cells)
        self.rnn_layer = tf.keras.layers.RNN(self.stacked_rnn_cells, return_sequences=True, return_state=True)

        self.fc1 = tf.keras.layers.Dense(self.units, activation='relu')
        self.fc2 = tf.keras.layers.Dense(self.vocab_size)

    def call(self, x, features, hidden_state):
        # 1. Attention
        if self.attention is not None:
            # Last layer state is the "representative" of the whole stack
            last_layer_state = hidden_state[-1][0] if self.cell_type == "LSTM" else hidden_state[-1]
            context_vector, attention_weights = self.attention(features, last_layer_state)
        else:
            context_vector = tf.reduce_mean(features, axis=1)
            attention_weights = None

        # 2. Concat Context with Embedding
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # 3. Process through stacked RNN
        rnn_results = self.rnn_layer(x, initial_state=hidden_state)
        output = rnn_results[0]
        next_state = rnn_results[1:]

        # 4. Dense heads
        x = self.fc1(output)
        x = tf.reshape(x, (-1, x.shape[2])) 
        logits = self.fc2(x)

        return logits, next_state, attention_weights

    def init_decoder_state(self, encoder_output):
        # Map (batch, 100, units) -> (batch, units)
        mean_features = tf.reduce_mean(encoder_output, axis=1)
        if self.cell_type == "LSTM":
            return [[mean_features, mean_features] for _ in range(self.num_layers)]
        else:
            return [mean_features for _ in range(self.num_layers)]
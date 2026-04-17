import tensorflow as tf
from src.models.attention import GlobalAttention, RegionAttention


class RNN_Decoder(tf.keras.Model):
    """
    Multi-layer RNN decoder with pluggable attention.
    Supports: LSTM / GRU cells, 1-N layers, Bahdanau / Dot / Scaled-Dot / RAN attention.
    Teacher Forcing is applied externally in the training loop (trainer.py).
    """

    def __init__(self, config):
        super(RNN_Decoder, self).__init__()
        self.units        = config['model']['units']
        self.vocab_size   = config['dataset']['vocab_size']
        self.embedding_dim = config['model']['embedding_dim']
        self.cell_type    = config['model']['decoder_type'].upper()
        self.num_layers   = config['model']['num_layers']
        self.attn_type    = config['model'].get('attention_type', 'None').lower()

        # --- Attention ---
        if self.attn_type == 'ran':
            self.attention = RegionAttention(config)
        elif self.attn_type in ('bahdanau', 'dot', 'scaled_dot'):
            self.attention = GlobalAttention(config)
        else:
            self.attention = None

        # --- Embedding ---
        self.embedding = tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim)

        # --- Stacked RNN (num_layers deep, each of size units) ---
        if self.cell_type == "LSTM":
            cells = [tf.keras.layers.LSTMCell(self.units) for _ in range(self.num_layers)]
        else:
            cells = [tf.keras.layers.GRUCell(self.units) for _ in range(self.num_layers)]

        self.stacked_rnn_cells = tf.keras.layers.StackedRNNCells(cells)
        self.rnn_layer = tf.keras.layers.RNN(
            self.stacked_rnn_cells, return_sequences=True, return_state=True
        )

        # --- Output heads ---
        self.fc1 = tf.keras.layers.Dense(self.units, activation='relu')
        self.fc2 = tf.keras.layers.Dense(self.vocab_size)

    def call(self, x, features, hidden_state):
        # --- 1. Attention context ---
        if self.attention is not None:
            last_h = hidden_state[-1][0] if self.cell_type == "LSTM" else hidden_state[-1]
            context_vector, attention_weights = self.attention(features, last_h)
        else:
            context_vector  = tf.reduce_mean(features, axis=1)
            attention_weights = None

        # --- 2. Embed token + concat context ---
        x = self.embedding(x)                                      # (batch, 1, embed_dim)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # --- 3. Stacked RNN ---
        rnn_results = self.rnn_layer(x, initial_state=hidden_state)
        output     = rnn_results[0]     # (batch, 1, units)
        next_state = rnn_results[1:]

        # --- 4. Dense heads ---
        out = self.fc1(output)
        out = tf.reshape(out, (-1, self.units))   # static units dim — always safe
        logits = self.fc2(out)                    # (batch, vocab_size)

        return logits, next_state, attention_weights

    def init_decoder_state(self, encoder_output):
        """Initialises every RNN layer's hidden state from the mean encoder feature."""
        mean_features = tf.reduce_mean(encoder_output, axis=1)   # (batch, units)
        if self.cell_type == "LSTM":
            return [[mean_features, mean_features] for _ in range(self.num_layers)]
        else:
            return [mean_features for _ in range(self.num_layers)]

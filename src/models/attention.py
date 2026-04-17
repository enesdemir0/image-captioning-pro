import tensorflow as tf


class RegionAttention(tf.keras.layers.Layer):
    """
    Region Attention Network (RAN).
    Applies a 3×3 spatial convolution over the 2-D feature grid before scoring,
    allowing the model to reason about local spatial regions rather than isolated
    point features.
    """

    def __init__(self, config):
        super(RegionAttention, self).__init__()
        self.units = config['model']['units']

        self.conv = tf.keras.layers.Conv2D(
            self.units, (3, 3), padding='same', activation='relu'
        )
        self.W = tf.keras.layers.Dense(self.units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features: (batch, num_features, units)
        batch_size   = tf.shape(features)[0]
        num_features = tf.shape(features)[1]

        # Infer square grid size dynamically
        grid_size = tf.cast(tf.math.sqrt(tf.cast(num_features, tf.float32)), tf.int32)

        # Reshape → 4-D for Conv2D  (batch, grid, grid, units)
        features_2d    = tf.reshape(features, (batch_size, grid_size, grid_size, self.units))
        region_features = self.conv(features_2d)

        # Flatten back  (batch, num_features, units)
        region_features = tf.reshape(region_features, (batch_size, num_features, self.units))

        # Standard additive attention score
        hidden_with_time = tf.expand_dims(hidden, 1)
        score = self.V(tf.nn.tanh(self.W(region_features) + self.W(hidden_with_time)))

        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector    = tf.reduce_sum(attention_weights * features, axis=1)

        return context_vector, attention_weights


class GlobalAttention(tf.keras.layers.Layer):
    """
    Three global attention flavours in one class:
      - bahdanau   : additive (Bahdanau et al. 2015)
      - dot        : Luong dot-product
      - scaled_dot : Transformer-style scaled dot-product
    """

    def __init__(self, config):
        super(GlobalAttention, self).__init__()
        self.units     = config['model']['units']
        self.attn_type = config['model'].get('attention_type', 'bahdanau').lower()

        if self.attn_type == 'bahdanau':
            self.W1 = tf.keras.layers.Dense(self.units)
            self.W2 = tf.keras.layers.Dense(self.units)
            self.V  = tf.keras.layers.Dense(1)
        elif self.attn_type == 'scaled_dot':
            self.dk = tf.cast(self.units, tf.float32)

    def call(self, features, hidden):
        hidden_with_time = tf.expand_dims(hidden, 1)   # (batch, 1, units)

        if self.attn_type == 'bahdanau':
            score = self.V(tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time)))
        elif self.attn_type == 'dot':
            score = tf.matmul(features, tf.transpose(hidden_with_time, perm=[0, 2, 1]))
        else:  # scaled_dot
            score = tf.matmul(
                features, tf.transpose(hidden_with_time, perm=[0, 2, 1])
            ) / tf.math.sqrt(self.dk)

        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector    = tf.reduce_sum(attention_weights * features, axis=1)

        return context_vector, attention_weights

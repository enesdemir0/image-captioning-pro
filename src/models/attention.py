import tensorflow as tf

class RegionAttention(tf.keras.layers.Layer):
    """
    Region Attention Network (RAN)
    Captures spatial dimensions by using a 2D Convolution inside the attention.
    """
    def __init__(self, config):
        super(RegionAttention, self).__init__()
        self.units = config['model']['units']
        
        # 2D convolution to look at 3x3 'regions' of pixels
        self.conv = tf.keras.layers.Conv2D(self.units, (3, 3), padding='same', activation='relu')
        self.W = tf.keras.layers.Dense(self.units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features input: (batch, num_features, units)
        batch_size = tf.shape(features)[0]
        num_features = tf.shape(features)[1]
        
        # Calculate grid size (e.g., 10 for Xception's 100 features)
        grid_size = tf.cast(tf.math.sqrt(tf.cast(num_features, tf.float32)), tf.int32)

        # 1. Reshape to 4D for Convolution (batch, grid, grid, units)
        features_2d = tf.reshape(features, (batch_size, grid_size, grid_size, self.units))

        # 2. Apply Spatial Convolution to capture 'Regions'
        region_features = self.conv(features_2d)
        
        # 3. Flatten back to (batch, num_features, units)
        region_features = tf.reshape(region_features, (batch_size, num_features, self.units))

        # 4. Standard Attention logic
        hidden_with_time = tf.expand_dims(hidden, 1)
        score = self.V(tf.nn.tanh(self.W(region_features) + self.W(hidden_with_time)))

        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class GlobalAttention(tf.keras.layers.Layer):
    """
    Supports: Bahdanau (Additive), Luong (Dot), and Transformer-style (Scaled Dot)
    """
    def __init__(self, config):
        super(GlobalAttention, self).__init__()
        self.units = config['model']['units']
        self.attn_type = config['model'].get('attention_type', 'bahdanau').lower()

        if self.attn_type == 'bahdanau':
            self.W1 = tf.keras.layers.Dense(self.units)
            self.W2 = tf.keras.layers.Dense(self.units)
            self.V = tf.keras.layers.Dense(1)
        elif self.attn_type == 'scaled_dot':
            self.dk = tf.cast(self.units, tf.float32)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        if self.attn_type == 'bahdanau':
            score = self.V(tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))
        elif self.attn_type == 'dot':
            score = tf.matmul(features, tf.transpose(hidden_with_time_axis, perm=[0, 2, 1]))
        elif self.attn_type == 'scaled_dot':
            score = tf.matmul(features, tf.transpose(hidden_with_time_axis, perm=[0, 2, 1])) / tf.math.sqrt(self.dk)

        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum(attention_weights * features, axis=1)

        return context_vector, attention_weights
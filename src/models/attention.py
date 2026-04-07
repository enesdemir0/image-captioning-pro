import tensorflow as tf


class RegionAttention(tf.keras.layers.Layer):
    """
    Bonus #2: Region Attention Network (RAN)
    Captures spatial dimensions by using a 2D Convolution inside the attention.
    """
    def __init__(self, config):
        super(RegionAttention, self).__init__()
        self.units = config['model']['units']
        
        # 2D convolution to look at 3x3 'regions'
        self.conv = tf.keras.layers.Conv2D(self.units, (3, 3), padding='same', activation='relu')
        self.W = tf.keras.layers.Dense(self.units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features input: (batch, 100, units)
        
        # --- THE FIX: Use static shape properties for the grid size ---
        num_features = features.shape[1] 
        grid_size = int(num_features**0.5) # Square root using Python (e.g., 10 for 100)

        # 1. Reshape to 4D for Convolution (batch, 10, 10, units)
        features_2d = tf.reshape(features, (-1, grid_size, grid_size, self.units))

        # 2. Apply Spatial Convolution to capture 'Regions'
        region_features = self.conv(features_2d)
        
        # 3. Flatten back to (batch, 100, units)
        region_features = tf.reshape(region_features, (-1, num_features, self.units))

        # 4. Standard Attention logic
        hidden_with_time = tf.expand_dims(hidden, 1)
        score = self.V(tf.nn.tanh(self.W(region_features) + self.W(hidden_with_time)))

        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class GlobalAttention(tf.keras.layers.Layer):
    """
    Implements 3 types of Attention Scoring Functions:
    1. Additive (Bahdanau)
    2. Dot-product (Luong)
    3. Scaled Dot-product (Transformer style)
    """
    def __init__(self, config):
        super(GlobalAttention, self).__init__()
        self.units = config['model']['units']
        self.attn_type = config['model'].get('attention_type', 'bahdanau').lower()

        # Layers for Bahdanau (Additive)
        if self.attn_type == 'bahdanau':
            self.W1 = tf.keras.layers.Dense(self.units)
            self.W2 = tf.keras.layers.Dense(self.units)
            self.V = tf.keras.layers.Dense(1)
        
        # Layers for Luong (Dot-product)
        elif self.attn_type == 'dot':
            # No extra layers needed, just matrix multiplication
            pass
            
        # Layers for Scaled Dot-product
        elif self.attn_type == 'scaled_dot':
            self.dk = tf.cast(self.units, tf.float32)

    def call(self, features, hidden):
        # features shape: (batch_size, 64, units)
        # hidden shape: (batch_size, units)

        if self.attn_type == 'bahdanau':
            # hidden_with_time_axis shape: (batch_size, 1, units)
            hidden_with_time_axis = tf.expand_dims(hidden, 1)
            # score shape: (batch_size, 64, 1)
            score = self.V(tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis)))

        elif self.attn_type == 'dot':
            hidden_with_time_axis = tf.expand_dims(hidden, 2)
            # score shape: (batch_size, 64, 1)
            score = tf.matmul(features, hidden_with_time_axis)

        elif self.attn_type == 'scaled_dot':
            hidden_with_time_axis = tf.expand_dims(hidden, 2)
            score = tf.matmul(features, hidden_with_time_axis) / tf.math.sqrt(self.dk)

        # attention_weights shape: (batch_size, 64, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum: (batch_size, units)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights
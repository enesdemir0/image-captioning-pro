import tensorflow as tf

class CNN_Encoder(tf.keras.Model):
    """
    Encoder that maps CNN features directly to the Decoder's hidden state size (units).
    This ensures the Attention mechanism math is always aligned.
    """
    def __init__(self, config):
        super(CNN_Encoder, self).__init__()
        self.encoder_name = config['model']['encoder_name']
        self.units = config['model']['units'] 

        # Dictionary-based loading makes it easier to support more models in the future
        self.model_map = {
            "InceptionV3": tf.keras.applications.InceptionV3,
            "VGG16": tf.keras.applications.VGG16,
            "Xception": tf.keras.applications.Xception
        }

        if self.encoder_name not in self.model_map:
            print(f"⚠️ Warning: {self.encoder_name} not in map, defaulting to Xception")
            base_model_builder = tf.keras.applications.Xception
        else:
            base_model_builder = self.model_map[self.encoder_name]

        # Load the base model without the classification head
        base_model = base_model_builder(include_top=False, weights='imagenet')

        # We take the output of the last convolutional layer
        self.feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.output)
        self.feature_extractor.trainable = False

        # The Bridge Layer: Maps CNN feature depth to the Decoder's 'units' (e.g., 512)
        self.fc = tf.keras.layers.Dense(self.units)

    def call(self, x):
        # 1. Extract features from CNN
        # Shape: (batch, grid_h, grid_w, channels)
        features = self.feature_extractor(x)
        
        # 2. Reshape to (batch, grid_size, channels)
        # For Xception, (10, 10, 2048) becomes (100, 2048)
        features = tf.reshape(features, (features.shape[0], -1, features.shape[3]))
        
        # 3. Pass through Dense layer to match Decoder Units
        # Shape: (batch, 100, 512)
        features = self.fc(features)
        
        # 4. Activation
        features = tf.nn.relu(features)
        
        return features
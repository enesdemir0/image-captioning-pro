import tensorflow as tf

class CNN_Encoder(tf.keras.Model):
    """
    The Encoder: Extracts features from an image using a pre-trained CNN.
    """
    def __init__(self, config):
        super(CNN_Encoder, self).__init__()
        self.encoder_name = config['model']['encoder_name']
        self.embedding_dim = config['model']['embedding_dim']

        # 1. Load the Base CNN (The "Eyes")
        if self.encoder_name == "InceptionV3":
            base_model = tf.keras.applications.InceptionV3(
                include_top=False, weights='imagenet'
            )
        elif self.encoder_name == "VGG16":
            base_model = tf.keras.applications.VGG16(
                include_top=False, weights='imagenet'
            )
        elif self.encoder_name == "Xception":
            base_model = tf.keras.applications.Xception(
                include_top=False, weights='imagenet'
            )
        else:
            raise ValueError(f"Unsupported encoder: {self.encoder_name}")

        # 2. Set the model to extract features from the last layer
        self.feature_extractor = tf.keras.Model(
            inputs=base_model.input, 
            outputs=base_model.layers[-1].output
        )
        
        # 3. Freeze the CNN (Professional Choice: We don't train the CNN)
        self.feature_extractor.trainable = False

        # 4. The "Bridge": A Dense layer to match the Decoder's embedding size
        self.fc = tf.keras.layers.Dense(self.embedding_dim)

    def call(self, x):
        # Extract features from CNN: (batch, height, width, channels)
        features = self.feature_extractor(x)
        
        # Flatten the spatial features: (batch, regions, channels)
        # This allows the Attention mechanism to look at specific "regions" later.
        features = tf.reshape(features, (features.shape[0], -1, features.shape[3]))
        
        # Pass through the "Bridge" (Dense layer)
        features = self.fc(features)
        features = tf.nn.relu(features)
        
        return features
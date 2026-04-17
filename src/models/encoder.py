import tensorflow as tf


class CNN_Encoder(tf.keras.Model):
    """
    Transfer-learning encoder. Extracts spatial feature maps from a frozen CNN
    backbone and projects them to the decoder's hidden-state dimension (units),
    so attention math is always dimension-aligned.
    """

    def __init__(self, config):
        super(CNN_Encoder, self).__init__()
        self.encoder_name = config['model']['encoder_name']
        self.units = config['model']['units']

        model_map = {
            "InceptionV3": tf.keras.applications.InceptionV3,
            "VGG16":       tf.keras.applications.VGG16,
            "Xception":    tf.keras.applications.Xception,
            "ResNet50":    tf.keras.applications.ResNet50,
        }

        if self.encoder_name not in model_map:
            print(f"Warning: '{self.encoder_name}' not supported — defaulting to Xception.")
            builder = tf.keras.applications.Xception
        else:
            builder = model_map[self.encoder_name]

        base = builder(include_top=False, weights='imagenet')
        self.feature_extractor = tf.keras.Model(inputs=base.input, outputs=base.output)
        self.feature_extractor.trainable = False

        # Bridge layer: maps CNN depth → decoder units (e.g., 2048 → 512)
        self.fc = tf.keras.layers.Dense(self.units, activation='relu')

    def call(self, x):
        # (batch, H, W, C)
        features = self.feature_extractor(x)

        # Dynamic reshape — safe for any batch size, including None at trace time
        batch = tf.shape(features)[0]
        c = tf.shape(features)[3]
        features = tf.reshape(features, (batch, -1, c))   # (batch, H*W, C)

        return self.fc(features)                           # (batch, H*W, units)

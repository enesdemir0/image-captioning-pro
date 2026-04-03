import tensorflow as tf

class CNN_Encoder(tf.keras.Model):
    def __init__(self, config):
        super(CNN_Encoder, self).__init__()
        self.encoder_name = config['model']['encoder_name']
        # CHANGE THIS LINE: use units, not embedding_dim
        self.units = config['model']['units'] 

        if self.encoder_name == "InceptionV3":
            base_model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
        elif self.encoder_name == "VGG16":
            base_model = tf.keras.applications.VGG16(include_top=False, weights='imagenet')
        else:
            base_model = tf.keras.applications.Xception(include_top=False, weights='imagenet')

        self.feature_extractor = tf.keras.Model(inputs=base_model.input, outputs=base_model.layers[-1].output)
        self.feature_extractor.trainable = False

        # FIX: This Dense layer MUST match the Decoder units (512)
        self.fc = tf.keras.layers.Dense(self.units)

    def call(self, x):
        features = self.feature_extractor(x)
        features = tf.reshape(features, (features.shape[0], -1, features.shape[3]))
        features = self.fc(features)
        features = tf.nn.relu(features)
        return features
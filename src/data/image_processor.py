import tensorflow as tf

class ImageProcessor:
    """
    Handles image loading and normalization for specific CNN encoders.
    """
    def __init__(self, config):
        self.encoder_name = config['model']['encoder_name']
        
        # Logic to set target size based on CNN requirements
        if self.encoder_name in ["InceptionV3", "Xception"]:
            self.target_size = (299, 299)
        elif self.encoder_name == "VGG16":
            self.target_size = (224, 224)
        else:
            self.target_size = (224, 224) # Default

    def preprocess_image(self, image_path):
        """Loads and prepares an image for the CNN."""
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.target_size)
        
        # Apply specific scaling for the chosen CNN
        if self.encoder_name == "InceptionV3":
            img = tf.keras.applications.inception_v3.preprocess_input(img)
        elif self.encoder_name == "VGG16":
            img = tf.keras.applications.vgg16.preprocess_input(img)
        elif self.encoder_name == "Xception":
            img = tf.keras.applications.xception.preprocess_input(img)
            
        return img, image_path
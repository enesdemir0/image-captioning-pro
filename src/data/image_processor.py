import tensorflow as tf

class ImageProcessor:
    """
    Professional Image Processor with dynamic model mapping.
    Ensures correct normalization for high-performance CNN Encoders.
    """
    def __init__(self, config):
        self.encoder_name = config['model']['encoder_name']
        
        # Configuration mapping for scalability
        self.model_configs = {
            "Xception": {"size": (299, 299), "func": tf.keras.applications.xception.preprocess_input},
            "InceptionV3": {"size": (299, 299), "func": tf.keras.applications.inception_v3.preprocess_input},
            "VGG16": {"size": (224, 224), "func": tf.keras.applications.vgg16.preprocess_input},
            "ResNet50": {"size": (224, 224), "func": tf.keras.applications.resnet50.preprocess_input}
        }
        
        # Get settings or default to VGG16 if unknown
        config_data = self.model_configs.get(self.encoder_name, self.model_configs["VGG16"])
        self.target_size = config_data["size"]
        self.preprocess_func = config_data["func"]

    def preprocess_image(self, image_path):
        """Loads, resizes, and normalizes image for the specific CNN backbone."""
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, self.target_size)
        
        # Apply the specific normalization (Xception needs [-1, 1], VGG needs Mean Subtraction)
        img = self.preprocess_func(img)
            
        return img, image_path
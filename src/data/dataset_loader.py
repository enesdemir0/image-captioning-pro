import json
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.data.text_processor import TextProcessor
from src.data.image_processor import ImageProcessor

class DataLoader:
    """
    Connects images and captions, manages Train/Val splits, 
    and creates high-performance tf.data.Datasets.
    """
    def __init__(self, config):
        self.config = config
        self.text_processor = TextProcessor(config)
        self.image_processor = ImageProcessor(config)
        
        self.image_dir = config['dataset']['image_dir']
        self.caption_file = config['dataset']['caption_file']
        self.image_prefix = config['dataset'].get('image_prefix', "") 

    def load_annotations(self):
        """Reads MS-COCO JSON and returns raw lists of image paths and captions."""
        with open(self.caption_file, 'r') as f:
            annotations = json.load(f)

        all_captions = []
        all_img_paths = []

        for ann in annotations['annotations']:
            caption = self.text_processor.clean_caption(ann['caption'])
            image_id = ann['image_id']
            
            img_name = f"{self.image_prefix}{str(image_id).zfill(12)}.jpg"
            full_image_path = os.path.join(self.image_dir, img_name)
            
            all_img_paths.append(full_image_path)
            all_captions.append(caption)

        return all_img_paths, all_captions

    def split_data(self, img_paths, captions):
        """
        Splits data into Training (80%) and Validation (20%) sets.
        Uses a fixed random_state for reproducibility.
        """
        return train_test_split(
            img_paths, 
            captions, 
            test_size=0.2, 
            random_state=42
        )

    def get_dataset(self, img_paths, captions, batch_size=64, is_training=True):
        """
        Creates a tf.data.Dataset.
        If is_training is True, it fits the tokenizer on these captions.
        """
        # 1. Tokenize the captions
        if is_training:
            # ONLY fit the tokenizer on training data
            self.text_processor.fit_on_texts(captions)
        
        cap_vector = self.text_processor.tokenize_and_pad(captions)

        # 2. Create the Dataset object
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, cap_vector))

        # 3. Map the image loading function (on-the-fly processing)
        dataset = dataset.map(
            lambda item1, item2: (self.image_processor.preprocess_image(item1)[0], item2),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # 4. Final pipeline steps
        if is_training:
            dataset = dataset.shuffle(1000)
            
        dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset
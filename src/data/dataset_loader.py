import json
import os
import pandas as pd
import tensorflow as tf
from src.data.text_processor import TextProcessor
from src.data.image_processor import ImageProcessor

class DataLoader:
    """
    Connects images and captions, and creates a tf.data.Dataset.
    """
    def __init__(self, config):
        self.config = config
        self.text_processor = TextProcessor(config)
        self.image_processor = ImageProcessor(config)
        self.image_dir = config['dataset']['image_dir']
        self.caption_file = config['dataset']['caption_file']

    def load_annotations(self):
        """Reads the MS-COCO JSON and returns a list of (image_path, caption)."""
        with open(self.caption_file, 'r') as f:
            annotations = json.load(f)

        all_captions = []
        all_img_paths = []

        # MS-COCO JSON structure: 'annotations' is a list of dicts
        for ann in annotations['annotations']:
            caption = self.text_processor.clean_caption(ann['caption'])
            image_id = ann['image_id']
            # Format the image ID to match the filename (e.g., COCO_train2014_000000031855.jpg)
            full_image_path = os.path.join(
                self.image_dir, 
                f"COCO_train2014_{str(image_id).zfill(12)}.jpg"
            )
            
            all_img_paths.append(full_image_path)
            all_captions.append(caption)

        return all_img_paths, all_captions

    def get_dataset(self, img_paths, captions, batch_size=64):
        """
        Creates a high-performance tf.data.Dataset.
        """
        # 1. Tokenize and pad the captions
        self.text_processor.fit_on_texts(captions)
        cap_vector = self.text_processor.tokenize_and_pad(captions)

        # 2. Create the Dataset object
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, cap_vector))

        # 3. Use the ImageProcessor to load images on-the-fly (Mapping)
        dataset = dataset.map(
            lambda item1, item2: (self.image_processor.preprocess_image(item1)[0], item2),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        # 4. Shuffle and batch
        dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        
        return dataset
import json
import os
import tensorflow as tf
from src.data.text_processor import TextProcessor
from src.data.image_processor import ImageProcessor

class DataLoader:
    def __init__(self, config):
        self.config = config
        self.text_processor = TextProcessor(config)
        self.image_processor = ImageProcessor(config)
        
        self.image_dir = config['dataset']['image_dir']
        self.caption_file = config['dataset']['caption_file']
        self.image_prefix = config['dataset'].get('image_prefix', "") 

    def load_annotations(self):
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

    def get_dataset(self, img_paths, captions, batch_size=64):
        self.text_processor.fit_on_texts(captions)
        cap_vector = self.text_processor.tokenize_and_pad(captions)
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, cap_vector))
        dataset = dataset.map(
            lambda item1, item2: (self.image_processor.preprocess_image(item1)[0], item2),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        dataset = dataset.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset
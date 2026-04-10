import json
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
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
        """Reads MS-COCO JSON and verifies every image file exists."""
        with open(self.caption_file, 'r') as f:
            annotations = json.load(f)
        
        all_captions, all_img_paths = [], []
        skipped_count = 0
        
        print(f"🔍 Verifying image files in {self.image_dir}...")
        for ann in annotations['annotations']:
            img_name = f"{self.image_prefix}{str(ann['image_id']).zfill(12)}.jpg"
            full_path = os.path.join(self.image_dir, img_name)
            
            # --- THE SAFETY CHECK ---
            if os.path.exists(full_path):
                caption = self.text_processor.clean_caption(ann['caption'])
                all_img_paths.append(full_path)
                all_captions.append(caption)
            else:
                skipped_count += 1

        print(f"✅ Verified {len(all_img_paths)} images. Skipped {skipped_count} missing files.")
        return all_img_paths, all_captions

    def split_data(self, img_paths, captions):
        """Creates a 70/15/15 split: Train, Val, and Test."""
        train_val_imgs, test_imgs, train_val_caps, test_caps = train_test_split(
            img_paths, captions, test_size=0.15, random_state=42
        )
        train_imgs, val_imgs, train_caps, val_caps = train_test_split(
            train_val_imgs, train_val_caps, test_size=0.176, random_state=42
        )
        return (train_imgs, train_caps), (val_imgs, val_caps), (test_imgs, test_caps)

    def get_dataset(self, img_paths, captions, batch_size=64, is_training=True):
        if is_training:
            self.text_processor.fit_on_texts(captions)
        
        cap_vector = self.text_processor.tokenize_and_pad(captions)
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, cap_vector))
        dataset = dataset.map(
            lambda i, c: (self.image_processor.preprocess_image(i)[0], c),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        if is_training:
            dataset = dataset.shuffle(1000)
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
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
        self.subset_size = config['dataset'].get('subset_size', 0)

    def load_annotations(self):
        """Reads MS-COCO JSON and verifies image integrity."""
        with open(self.caption_file, 'r') as f:
            annotations = json.load(f)
        
        all_captions, all_img_paths = [], []
        
        print(f"🔍 Loading annotations...")
        # If subset_size is > 0, we only take that many annotations
        anns = annotations['annotations']
        if self.subset_size > 0:
            anns = anns[:self.subset_size]
            print(f"🧪 Subsetting data to {self.subset_size} samples for fast testing.")

        for ann in anns:
            img_name = f"{self.image_prefix}{str(ann['image_id']).zfill(12)}.jpg"
            full_path = os.path.join(self.image_dir, img_name)
            
            if os.path.exists(full_path):
                caption = self.text_processor.clean_caption(ann['caption'])
                all_img_paths.append(full_path)
                all_captions.append(caption)

        print(f"✅ Verified {len(all_img_paths)} images.")
        return all_img_paths, all_captions

    def prepare_datasets(self, batch_size=64):
        """High-level method to load, split, fit tokenizer, and return all datasets."""
        img_paths, captions = self.load_annotations()
        
        # 1. Split Data (70/15/15)
        (train_imgs, train_caps), (val_imgs, val_caps), (test_imgs, test_caps) = self.split_data(img_paths, captions)
        
        # 2. IMPORTANT: Fit tokenizer ONLY on the training captions
        print("🔡 Fitting tokenizer on training data...")
        self.text_processor.fit_on_texts(train_caps)
        
        # 3. Create tf.data objects
        train_ds = self.get_dataset(train_imgs, train_caps, batch_size, is_training=True)
        val_ds = self.get_dataset(val_imgs, val_caps, batch_size, is_training=False)
        test_ds = self.get_dataset(test_imgs, test_caps, batch_size, is_training=False)
        
        return train_ds, val_ds, test_ds

    def split_data(self, img_paths, captions):
        """Creates professional Train, Val, and Test splits."""
        train_val_imgs, test_imgs, train_val_caps, test_caps = train_test_split(
            img_paths, captions, test_size=0.15, random_state=42
        )
        train_imgs, val_imgs, train_caps, val_caps = train_test_split(
            train_val_imgs, train_val_caps, test_size=0.176, random_state=42
        )
        return (train_imgs, train_caps), (val_imgs, val_caps), (test_imgs, test_caps)

    def get_dataset(self, img_paths, captions, batch_size=64, is_training=True):
        """Converts raw paths and text into a high-performance streaming pipeline."""
        cap_vector = self.text_processor.tokenize_and_pad(captions)
        
        dataset = tf.data.Dataset.from_tensor_slices((img_paths, cap_vector))
        
        # Map image loading (index [0] because preprocess_image returns img, path)
        dataset = dataset.map(
            lambda i, c: (self.image_processor.preprocess_image(i)[0], c),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        if is_training:
            # Buffer size 10,000 is better for larger datasets like COCO
            dataset = dataset.shuffle(10000).repeat() 
        
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
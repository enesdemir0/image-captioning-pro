"""
Run this ONCE after setting up the MS-COCO data.
It reproduces the exact same split logic as the main branch (random_state=42,
test_size=0.15) and picks the same 300 samples (seed=42), then saves them to
configs/test_split.json.

Captions are cleaned with the same logic as main branch text_processor.clean_caption()
so reference text used for BLEU/METEOR/ROUGE-L/BERTScore is identical to main branch.

Commit that file to git so every VLM experiment evaluates on the same images.
"""
import json
import os
import re
import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.config_loader import load_config


def clean_caption(caption):
    """Mirrors main branch src/data/text_processor.py clean_caption() exactly."""
    caption = str(caption).lower()
    caption = re.sub(r'[^\w\s]', '', caption)
    return re.sub(r'\s+', ' ', caption).strip()


def main():
    config = load_config()
    caption_file = config['dataset']['caption_file']
    image_dir = config['dataset']['image_dir']
    image_prefix = config['dataset'].get('image_prefix', '')
    num_samples = config['evaluation']['num_samples']
    random_seed = config['evaluation']['random_seed']
    output_path = config['dataset']['test_split_path']

    print("Loading MS-COCO annotations...")
    with open(caption_file, 'r') as f:
        annotations = json.load(f)

    img_paths, captions = [], []
    for ann in annotations['annotations']:
        img_name = f"{image_prefix}{str(ann['image_id']).zfill(12)}.jpg"
        full_path = os.path.join(image_dir, img_name)
        if os.path.exists(full_path):
            img_paths.append(full_path)
            captions.append(clean_caption(ann['caption']))

    print(f"Found {len(img_paths)} image-caption pairs.")

    # Mirrors main branch split_data() exactly
    _, test_imgs, _, test_caps = train_test_split(
        img_paths, captions, test_size=0.15, random_state=42
    )

    # Mirrors main branch evaluate.py sampling exactly
    indices = np.arange(len(test_imgs))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    selected = indices[:num_samples]

    samples = [
        {"image_path": test_imgs[i], "caption": test_caps[i]}
        for i in selected
    ]

    with open(output_path, 'w') as f:
        json.dump({
            "num_samples": num_samples,
            "random_seed": random_seed,
            "split_random_state": 42,
            "test_size": 0.15,
            "samples": samples
        }, f, indent=2)

    print(f"Saved {num_samples} test samples → {output_path}")
    print("Commit configs/test_split.json to git before running any VLM evaluation.")


if __name__ == "__main__":
    main()

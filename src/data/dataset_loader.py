import json
import os


class COCODataLoader:
    def __init__(self, config):
        self.config = config
        self.image_dir = config['dataset']['image_dir']
        self.caption_file = config['dataset']['caption_file']
        self.image_prefix = config['dataset'].get('image_prefix', '')
        self.test_split_path = config['dataset']['test_split_path']

    def load_test_split(self):
        """
        Load the fixed test split and return the first num_samples entries.

        Slicing the head (not re-shuffling) guarantees that every model
        evaluated with the same num_samples sees the exact same images,
        regardless of which model runs first.
        """
        if not os.path.exists(self.test_split_path):
            raise FileNotFoundError(
                f"Test split not found at '{self.test_split_path}'. "
                "Run scripts/generate_test_split.py first, then commit the file."
            )
        with open(self.test_split_path, 'r') as f:
            data = json.load(f)

        samples = data['samples']
        num_samples = self.config['evaluation'].get('num_samples', len(samples))
        return samples[:num_samples]

    def load_few_shot_examples(self):
        """
        Load the pool of in-context (image, caption) examples used by the
        few_shot strategy. Drawn from the train portion of the split by
        generate_test_split.py, so they never overlap the held-out test set.
        """
        if not os.path.exists(self.test_split_path):
            raise FileNotFoundError(
                f"Test split not found at '{self.test_split_path}'. "
                "Run scripts/generate_test_split.py first, then commit the file."
            )
        with open(self.test_split_path, 'r') as f:
            data = json.load(f)

        examples = data.get('few_shot_examples')
        if not examples:
            raise ValueError(
                f"'{self.test_split_path}' has no 'few_shot_examples' — it was likely "
                "generated before few_shot support was added. Delete it and rerun "
                "scripts/generate_test_split.py to regenerate it with examples included."
            )

        k = self.config['model'].get('few_shot_k', len(examples))
        return examples[:k]

    def load_all_annotations(self):
        """Load raw COCO annotations — used only by generate_test_split.py."""
        with open(self.caption_file, 'r') as f:
            annotations = json.load(f)

        img_paths, captions = [], []
        subset_size = self.config['dataset'].get('subset_size', 0)
        anns = annotations['annotations']
        if subset_size > 0:
            anns = anns[:subset_size]

        for ann in anns:
            img_name = f"{self.image_prefix}{str(ann['image_id']).zfill(12)}.jpg"
            full_path = os.path.join(self.image_dir, img_name)
            if os.path.exists(full_path):
                img_paths.append(full_path)
                captions.append(ann['caption'])

        return img_paths, captions

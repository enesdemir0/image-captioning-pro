from src.utils.config_loader import load_config
from src.data.dataset_loader import DataLoader

# 1. Load the config
config = load_config()

# 2. Initialize the DataLoader
loader = DataLoader(config)

# 3. Try to load the annotations
print("Reading MS-COCO Annotations... please wait.")
img_paths, captions = loader.load_annotations()

# 4. Check the results
print("-" * 30)
print(f"SUCCESS! Total captions found: {len(captions)}")
print(f"Example Image Path: {img_paths[0]}")
print(f"Example Caption: {captions[0]}")
print("-" * 30)
import os
import requests
import zipfile
from tqdm import tqdm
from src.utils.config_loader import load_config

def download_file(url, destination):
    """Downloads a file with a progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=destination,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def setup_data():
    # 1. Load the config (The "Brain")
    config = load_config()
    
    # 2. Get the URL and Paths from the config
    url_ann = config['dataset']['url_annotations'] # NO LONGER HARDCODED
    caption_file_path = config['dataset']['caption_file']
    base_data_dir = "data" 

    # Determine the directory where the json should live
    annotations_dir = os.path.dirname(caption_file_path)
    os.makedirs(annotations_dir, exist_ok=True)

    # 3. Download and Extract logic
    ann_zip_path = os.path.join(base_data_dir, "annotations.zip")
    
    # Check if the caption file already exists
    if not os.path.exists(caption_file_path):
        print(f"Downloading from: {url_ann}")
        download_file(url_ann, ann_zip_path)
        
        print("Extracting files...")
        with zipfile.ZipFile(ann_zip_path, 'r') as zip_ref:
            zip_ref.extractall(base_data_dir)
        
        os.remove(ann_zip_path)
        print("Done!")
    else:
        print("Data already exists at the path specified in config. Skipping.")

if __name__ == "__main__":
    setup_data()
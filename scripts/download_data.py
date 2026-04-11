import os
import requests
import zipfile
from tqdm import tqdm
from pathlib import Path
from src.utils.config_loader import load_config

def download_file(url, destination):
    """Downloads a file with a professional progress bar and error handling."""
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status() # Check for 404/500 errors
        total_size = int(response.headers.get('content-length', 0))
        
        with open(destination, 'wb') as file, tqdm(
            desc=f"📥 Downloading {os.path.basename(destination)}",
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=8192): # Slightly larger chunks for speed
                size = file.write(data)
                bar.update(size)
    except Exception as e:
        print(f"❌ Error downloading {url}: {e}")
        if os.path.exists(destination):
            os.remove(destination)
        return False
    return True

def setup_data():
    # 1. Load config
    config = load_config()
    
    # 2. Get Paths
    url_ann = config['dataset']['url_annotations']
    caption_file_path = config['dataset']['caption_file']
    base_data_dir = Path("data")
    base_data_dir.mkdir(exist_ok=True)

    # Path to store the annotations zip
    ann_zip_path = base_data_dir / "annotations.zip"
    
    # --- ANNOTATIONS LOGIC ---
    if not os.path.exists(caption_file_path):
        print(f"🚀 Annotations not found. Starting download...")
        success = download_file(url_ann, ann_zip_path)
        
        if success:
            print("📦 Extracting annotations...")
            with zipfile.ZipFile(ann_zip_path, 'r') as zip_ref:
                zip_ref.extractall(base_data_dir)
            
            os.remove(ann_zip_path)
            print("✨ Annotations ready!")
    else:
        print("✅ Annotations already exist. Skipping.")

    # --- IMAGE DIRECTORY CHECK ---
    image_dir = config['dataset']['image_dir']
    if not os.path.exists(image_dir):
        print(f"⚠️  WARNING: Image directory '{image_dir}' not found!")
        print(f"Please ensure you have downloaded the MS-COCO images and placed them in {image_dir}")
        # Option: You could add a download_file(url_images, ...) here, 
        # but 13GB is a lot to download automatically!

if __name__ == "__main__":
    setup_data()
import os
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

from src.utils.config_loader import load_config


def download_file(url, destination):
    try:
        response = requests.get(url, stream=True, timeout=10)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        with open(destination, 'wb') as f, tqdm(
            desc=f"Downloading {os.path.basename(destination)}",
            total=total_size, unit='iB', unit_scale=True, unit_divisor=1024
        ) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                bar.update(f.write(chunk))
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(destination):
            os.remove(destination)
        return False
    return True


def main():
    config = load_config()
    url_ann = config['dataset']['url_annotations']
    caption_file = config['dataset']['caption_file']
    base_data_dir = Path("data")
    base_data_dir.mkdir(exist_ok=True)
    ann_zip = base_data_dir / "annotations.zip"

    if not os.path.exists(caption_file):
        print("Downloading MS-COCO annotations...")
        if download_file(url_ann, ann_zip):
            print("Extracting...")
            with zipfile.ZipFile(ann_zip, 'r') as z:
                z.extractall(base_data_dir)
            os.remove(ann_zip)
            print("Annotations ready.")
    else:
        print("Annotations already exist. Skipping.")

    image_dir = config['dataset']['image_dir']
    if not os.path.exists(image_dir):
        print(f"WARNING: Image directory '{image_dir}' not found.")
        print("Unzip coco_train2014.zip from your Google Drive into data/")


if __name__ == "__main__":
    main()

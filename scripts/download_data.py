import os
import requests
import zipfile
from tqdm import tqdm

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
    # 1. Create directory structure
    base_dir = "data"
    annotations_dir = os.path.join(base_dir, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)

    # 2. URLs for MS-COCO 2014
    urls = {
        "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
        # We will add images later if needed
    }

    # 3. Download and Extract Annotations
    ann_zip = os.path.join(base_dir, "annotations.zip")
    if not os.path.exists(os.path.join(annotations_dir, "captions_train2014.json")):
        print("Downloading MS-COCO Annotations...")
        download_file(urls["annotations"], ann_zip)
        
        print("Extracting Annotations...")
        with zipfile.ZipFile(ann_zip, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
        os.remove(ann_zip) # Clean up zip file
        print("Annotations ready!")
    else:
        print("Annotations already exist. Skipping.")

if __name__ == "__main__":
    setup_data()
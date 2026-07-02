import os
import subprocess

# Always resolve paths relative to the project root, not wherever Colab's CWD happens to be
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(PROJECT_ROOT)

DRIVE_ZIP    = "/content/drive/MyDrive/coco_train2014.zip"
COCO_IMG_URL = "http://images.cocodataset.org/zips/train2014.zip"
LOCAL_ZIP    = "/content/coco_train2014.zip"
DATA_DIR     = os.path.join(PROJECT_ROOT, "data", "train2014")


def run(cmd):
    """Run a shell command with PYTHONPATH always pointing to the project root."""
    print(f"\nExecuting: {cmd}")
    env = os.environ.copy()
    env['PYTHONPATH'] = PROJECT_ROOT
    subprocess.run(cmd, shell=True, check=True, env=env)


def main():
    # 1. Dependencies
    run("pip install -e . -q")
    run("pip install transformers --upgrade -q")
    run("pip install accelerate bitsandbytes sentencepiece "
        "bert-score dagshub mlflow nltk rouge-score scikit-learn tqdm pillow pyyaml -q")

    # 2. DagsHub auth
    import dagshub
    dagshub.init(repo_owner="enesdemir0", repo_name="image-captioning-pro", mlflow=True)

    # 3. Images — Drive zip → direct COCO download → already extracted
    os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
    if os.path.exists(DATA_DIR):
        print("Images already present on local disk. Skipping.")
    elif os.path.exists(DRIVE_ZIP):
        print("Extracting images from Google Drive zip...")
        run(f"unzip -q {DRIVE_ZIP} -d data/")
    else:
        print("Images not found in Drive. Downloading directly from MS-COCO (~13 GB)...")
        print("This takes ~20-40 min on Colab. Go get a coffee.")
        run(f"wget -q --show-progress -O {LOCAL_ZIP} {COCO_IMG_URL}")
        print("Extracting...")
        run(f"unzip -q {LOCAL_ZIP} -d data/")
        os.remove(LOCAL_ZIP)
        print("\nTip: save to Drive so you never download again:")
        print("  !cd data && zip -r /content/drive/MyDrive/coco_train2014.zip train2014")

    # 4. Annotations (fast, ~250 MB)
    run("python scripts/download_data.py")

    # 5. Fixed test split — generate once, then committed to git
    if not os.path.exists("configs/test_split.json"):
        print("Generating fixed test split (runs once)...")
        run("python scripts/generate_test_split.py")
        print(">>> Commit configs/test_split.json to git before switching models. <<<")
    else:
        print("Fixed test split found.")

    # 6. Evaluate the model defined in vlm_config.yaml
    run("python scripts/evaluate.py")

    print("\nDone! View results at https://dagshub.com/enesdemir0/image-captioning-pro")


if __name__ == "__main__":
    main()

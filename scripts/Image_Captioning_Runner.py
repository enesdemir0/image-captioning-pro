import os
import sys

def run_command(cmd):
    print(f"Executing: {cmd}")
    os.system(cmd)

def main():
    # 1. MOUNT GOOGLE DRIVE
    print("📂 Mounting Google Drive...")
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except ImportError:
        print("⚠️ Not running in Google Colab. Skipping Drive mount.")

    # 2. INSTALL DEPENDENCIES
    print("🛠 Installing dependencies and project structure...")
    run_command("pip install -e . -q")
    run_command("pip install dagshub mlflow nltk rouge-score opencv-python tqdm -q")

    # 3. DATA INFRASTRUCTURE
    print("📥 Fetching MS-COCO annotations...")
    run_command("python scripts/download_data.py")

    ZIP_PATH = "/content/drive/MyDrive/coco_train2014.zip"
    if not os.path.exists("data/train2014"):
        print("📦 Extracting 13GB dataset to local disk for L4 GPU speed...")
        os.makedirs("data", exist_ok=True)
        run_command(f"unzip -q {ZIP_PATH} -d data/")
        print("✅ Dataset Ready!")
    else:
        print("✅ Dataset already exists on local disk.")

    # 4. DAGSHUB / MLFLOW AUTH
    print("🔐 Initializing DagsHub...")
    import dagshub
    dagshub.init(repo_owner="enesdemir0", repo_name="image-captioning-pro", mlflow=True)

    # 5. EXECUTION PIPELINE
    # Set PYTHONPATH so 'src' is visible
    os.environ['PYTHONPATH'] = os.getcwd()

    print("\n🚀 STEP 1: STARTING TRAINING...")
    run_command("python src/training/train.py")

    print("\n🔥 STEP 2: STARTING EVALUATION & INTERPRETABILITY HEATMAPS...")
    run_command("python scripts/evaluate.py")

    print("\n✨ PIPELINE COMPLETE! View results on DagsHub and in results/samples/")

if __name__ == "__main__":
    main()
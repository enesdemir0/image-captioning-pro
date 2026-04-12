import os
import subprocess
import sys

def run_command(cmd):
    print(f"\n🏃 Executing: {cmd}")
    # Using subprocess.run ensures the script waits for the command to finish
    subprocess.run(cmd, shell=True, check=True)

def main():
    # 1. INSTALL DEPENDENCIES
    print("🛠 Installing dependencies and project structure...")
    run_command("pip install -e . -q")
    run_command("pip install dagshub mlflow nltk rouge-score opencv-python tqdm -q")

    # 2. DATA INFRASTRUCTURE
    print("📥 Fetching MS-COCO annotations...")
    run_command("python scripts/download_data.py")

    # The images unzipping logic
    ZIP_PATH = "/content/drive/MyDrive/coco_train2014.zip"
    if not os.path.exists("data/train2014"):
        print("📦 Extracting 13GB dataset to local disk for speed...")
        os.makedirs("data", exist_ok=True)
        run_command(f"unzip -q {ZIP_PATH} -d data/")
        print("✅ Dataset Ready!")
    else:
        print("✅ Dataset already exists on local disk.")

    # 3. DAGSHUB / MLFLOW AUTH
    print("🔐 Initializing DagsHub...")
    import dagshub
    # This will trigger the link if not already authorized
    dagshub.init(repo_owner="enesdemir0", repo_name="image-captioning-pro", mlflow=True)

    # 4. EXECUTION PIPELINE
    # Set PYTHONPATH so 'src' is visible to the sub-scripts
    os.environ['PYTHONPATH'] = os.getcwd()

    print("\n🚀 STEP 1: STARTING TRAINING...")
    run_command("python src/training/train.py")

    print("\n🔥 STEP 2: STARTING EVALUATION & INTERPRETABILITY HEATMAPS...")
    run_command("python scripts/evaluate.py")

    print("\n✨ PIPELINE COMPLETE! View results on DagsHub and in results/samples/")

if __name__ == "__main__":
    main()
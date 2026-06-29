import os
import subprocess
import sys
import yaml

def run_command(cmd):
    print(f"\n🏃 Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    # 1. READ CONFIG FIRST (needed for zip_path and mode)
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    mode     = config.get('mode', 'cnn_rnn')
    ZIP_PATH = config['dataset'].get('zip_path', '/content/drive/MyDrive/coco_train2014.zip')

    # 2. INSTALL DEPENDENCIES
    print("🛠 Installing dependencies and project structure...")
    run_command("pip install -e . -q")
    run_command("pip install dagshub mlflow nltk rouge-score opencv-python tqdm -q")

    # 3. DATA INFRASTRUCTURE
    print("📥 Fetching MS-COCO annotations...")
    run_command("python scripts/download_data.py")

    if mode == "vit_gpt2":
        # Images are fetched on-the-fly from COCO URLs inside evaluate_vit_gpt2.py
        # so no local zip or image directory is required.
        if os.path.exists("data/train2014"):
            print("✅ Local images found — will use disk cache.")
        else:
            print("ℹ️  No local images — evaluate_vit_gpt2 will fetch them via COCO URLs.")
    else:
        if not os.path.exists("data/train2014"):
            if not os.path.exists(ZIP_PATH):
                print(f"\n❌ COCO images not found at data/train2014")
                print(f"   Zip not found at: {ZIP_PATH}")
                print(f"   Update 'zip_path' in configs/config.yaml or download the dataset.")
                sys.exit(1)
            print("📦 Extracting dataset to local disk for speed...")
            os.makedirs("data", exist_ok=True)
            run_command(f"unzip -q \"{ZIP_PATH}\" -d data/")
            print("✅ Dataset Ready!")
        else:
            print("✅ Dataset already exists on local disk.")

    # 4. DAGSHUB / MLFLOW AUTH
    print("🔐 Initializing DagsHub...")
    import dagshub
    dagshub.init(repo_owner="enesdemir0", repo_name="image-captioning-pro", mlflow=True)

    # 5. EXECUTION PIPELINE
    os.environ['PYTHONPATH'] = os.getcwd()
    print(f"\n⚙️  Mode: {mode}")

    if mode == "vit_gpt2":
        print("🤖 ViT-GPT2 (Pre-trained) — skipping training.")
        run_command("pip install torch transformers -q")

        print("\n🔥 STARTING ViT-GPT2 EVALUATION...")
        run_command("python scripts/evaluate_vit_gpt2.py")

    else:
        print("🚀 CNN+RNN (Custom Training)")

        print("\n🚀 STEP 1: STARTING TRAINING...")
        run_command("python src/training/train.py")

        print("\n🔥 STEP 2: STARTING EVALUATION & INTERPRETABILITY HEATMAPS...")
        run_command("python scripts/evaluate.py")

    print("\n✨ PIPELINE COMPLETE! View results on DagsHub and in results/")

if __name__ == "__main__":
    main()
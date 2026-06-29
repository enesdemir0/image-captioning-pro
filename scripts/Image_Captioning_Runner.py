import os
import subprocess
import sys
import yaml

def run_command(cmd):
    print(f"\n🏃 Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    # 1. INSTALL DEPENDENCIES
    print("🛠 Installing dependencies and project structure...")
    run_command("pip install -e . -q")
    run_command("pip install dagshub mlflow nltk rouge-score opencv-python tqdm -q")

    # 2. DATA INFRASTRUCTURE
    print("📥 Fetching MS-COCO annotations...")
    run_command("python scripts/download_data.py")

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
    dagshub.init(repo_owner="enesdemir0", repo_name="image-captioning-pro", mlflow=True)

    # 4. EXECUTION PIPELINE
    os.environ['PYTHONPATH'] = os.getcwd()

    # Read mode from config (default: cnn_rnn)
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    mode = config.get('mode', 'cnn_rnn')

    if mode == "vit_gpt2":
        print("\n🤖 MODE: ViT-GPT2 (Pre-trained) — skipping training step.")
        run_command("pip install torch transformers -q")

        print("\n🔥 STARTING ViT-GPT2 EVALUATION...")
        run_command("python scripts/evaluate_vit_gpt2.py")

    else:
        print("\n🚀 MODE: CNN+RNN (Custom Training)")

        print("\n🚀 STEP 1: STARTING TRAINING...")
        run_command("python src/training/train.py")

        print("\n🔥 STEP 2: STARTING EVALUATION & INTERPRETABILITY HEATMAPS...")
        run_command("python scripts/evaluate.py")

    print("\n✨ PIPELINE COMPLETE! View results on DagsHub and in results/")

if __name__ == "__main__":
    main()
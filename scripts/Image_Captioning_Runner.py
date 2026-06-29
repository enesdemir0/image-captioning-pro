import os
import subprocess
import sys
import argparse


def run_command(cmd):
    print(f"\n🏃 Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Config file to use. Pass configs/config_dev.yaml for the 20-image sample run.",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Shorthand for --config configs/config_dev.yaml",
    )
    args = parser.parse_args()

    config_path = "configs/config_dev.yaml" if args.dev else args.config

    # 1. INSTALL DEPENDENCIES
    print("🛠 Installing dependencies and project structure...")
    run_command("pip install -e . -q")
    run_command("pip install dagshub mlflow nltk rouge-score opencv-python tqdm -q")

    # 2. DATA INFRASTRUCTURE
    if config_path == "configs/config_dev.yaml":
        # Dev mode: pull the 20-image sample from DVC instead of unzipping 13GB
        print("📥 Pulling sample dataset from DVC...")
        run_command("dvc pull")
    else:
        # Production mode: annotations from download_data.py + images from Drive zip
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

    print("\n🚀 STEP 1: STARTING TRAINING...")
    run_command(f"python src/training/train.py --config {config_path}")

    print("\n🔥 STEP 2: STARTING EVALUATION & INTERPRETABILITY HEATMAPS...")
    run_command(f"python scripts/evaluate.py --config {config_path}")

    print("\n✨ PIPELINE COMPLETE! View results on DagsHub and in results/samples/")


if __name__ == "__main__":
    main()
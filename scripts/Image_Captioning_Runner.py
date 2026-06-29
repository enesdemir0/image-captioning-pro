import os
import subprocess


def run_command(cmd):
    print(f"\n🏃 Executing: {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def main():
    # 1. INSTALL DEPENDENCIES
    print("🛠️  Installing dependencies...")
    run_command("pip install -e . -q")
    run_command("pip install dvc[http] dagshub mlflow nltk rouge-score opencv-python tqdm -q")

    # 2. AUTHENTICATE WITH DAGSHUB (needed for both DVC pull and MLflow)
    print("🔐 Authenticating with DagsHub...")
    import dagshub
    dagshub.init(repo_owner="enesdemir0", repo_name="image-captioning-pro", mlflow=True)

    # After dagshub.init(), set DVC credentials so dvc pull works without a browser
    token = os.environ.get("DAGSHUB_USER_TOKEN", "")
    if token:
        run_command("dvc remote modify myremote --local auth basic")
        run_command("dvc remote modify myremote --local user enesdemir0")
        run_command(f"dvc remote modify myremote --local password {token}")

    # 3. PULL SAMPLE DATASET FROM DVC
    print("📥 Pulling sample dataset from DagsHub DVC storage...")
    run_command("dvc pull")

    # 4. RUN PIPELINE ON SAMPLE DATA
    os.environ["PYTHONPATH"] = os.getcwd()

    print("\n🚀 STEP 1: TRAINING on sample dataset...")
    run_command("python -m src.training.train --config configs/config_dev.yaml")

    print("\n🔥 STEP 2: EVALUATION...")
    run_command("python scripts/evaluate.py --config configs/config_dev.yaml")

    print("\n✨ PIPELINE COMPLETE! View results on DagsHub MLflow.")


if __name__ == "__main__":
    main()

import json
import os
import mlflow
import dagshub
import nltk
from tqdm import tqdm

from src.utils.config_loader import load_config
from src.utils.metrics import calculate_metrics
from src.data.dataset_loader import COCODataLoader
from src.models import get_vlm

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


def main():
    config = load_config()

    dagshub.init(
        repo_owner=config['mlflow']['repo_owner'],
        repo_name=config['mlflow']['repo_name'],
        mlflow=True
    )
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])

    vlm = get_vlm(config)
    model_id = vlm.get_model_id()
    print(f"Model ID: {model_id}")
    mlflow.set_experiment(model_id)

    loader = COCODataLoader(config)
    samples = loader.load_test_split()
    print(f"Evaluating on {len(samples)} fixed test samples.")

    vlm.load()

    references, hypotheses = [], []
    os.makedirs("results", exist_ok=True)

    with mlflow.start_run(run_name="VLM_Evaluation"):
        mlflow.log_params(config['model'])
        mlflow.log_param("num_samples", len(samples))

        for i, sample in enumerate(tqdm(samples, desc="Generating captions")):
            caption = vlm.generate_caption(sample['image_path'])
            references.append(sample['caption'])
            hypotheses.append(caption)

            if i % 50 == 0:
                print(f"  [{i}/{len(samples)}] PRED: {caption}")

        metrics = calculate_metrics(references, hypotheses)
        mlflow.log_metrics(metrics)
        print(f"\nResults:\n{json.dumps(metrics, indent=2)}")

        summary_path = f"results/{model_id}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(summary_path)


if __name__ == "__main__":
    main()

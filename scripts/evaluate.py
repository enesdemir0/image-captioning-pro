import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import mlflow
import dagshub
import nltk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


def save_sample_image(image_path, real_caption, pred_caption, output_path):
    img = plt.imread(image_path)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(
        f"REAL:  {real_caption}\n\nPRED:  {pred_caption}",
        fontsize=10, pad=15, loc='left', wrap=True
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


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

    if config['model'].get('strategy', 'zero_shot') == 'few_shot':
        vlm.few_shot_examples = loader.load_few_shot_examples()
        print(f"Using {len(vlm.few_shot_examples)} few-shot examples.")

    vlm.load()

    references, hypotheses = [], []
    os.makedirs("results/samples", exist_ok=True)
    num_saved = 0

    with mlflow.start_run(run_name="VLM_Evaluation"):
        mlflow.log_params(config['model'])
        mlflow.log_param("num_samples", len(samples))

        for i, sample in enumerate(tqdm(samples, desc="Generating captions")):
            caption = vlm.generate_caption(sample['image_path'])
            references.append(sample['caption'])
            hypotheses.append(caption)

            if num_saved < 5:
                out_path = f"results/samples/sample_{i}.png"
                save_sample_image(sample['image_path'], sample['caption'], caption, out_path)
                mlflow.log_artifact(out_path, artifact_path="samples")
                num_saved += 1

            if i % 50 == 0:
                print(f"  [{i}/{len(samples)}] PRED: {caption}")

        vlm.unload()
        metrics = calculate_metrics(references, hypotheses)
        mlflow.log_metrics(metrics)
        print(f"\nResults:\n{json.dumps(metrics, indent=2)}")

        summary_path = f"results/{model_id}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(summary_path)


if __name__ == "__main__":
    main()

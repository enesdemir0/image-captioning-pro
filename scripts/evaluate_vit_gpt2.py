import os
import io
import json
import re
import numpy as np
import requests
import mlflow
import dagshub
from PIL import Image
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk

from src.utils.config_loader import load_config
from src.models.vit_gpt2 import ViTGPT2Captioner

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)

_COCO_URL = "http://images.cocodataset.org/train2014/{filename}"


def load_image(img_path: str, filename: str) -> Image.Image:
    """Load from local disk; fall back to COCO public URL if not found."""
    if os.path.exists(img_path):
        return Image.open(img_path).convert("RGB")
    url = _COCO_URL.format(filename=filename)
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def clean_caption(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()


def build_test_pairs(config, n_eval):
    """Read COCO annotations directly and return sampled test pairs.

    Bypasses DataLoader so missing local images are not filtered out —
    load_image() will fetch them from COCO URLs instead.
    """
    with open(config['dataset']['caption_file']) as f:
        raw = json.load(f)

    img_map = {img['id']: img['file_name'] for img in raw['images']}
    image_dir = config['dataset']['image_dir']

    pairs = []
    for ann in raw['annotations']:
        filename = img_map.get(ann['image_id'])
        if not filename:
            continue
        pairs.append((
            os.path.join(image_dir, filename),   # local path (may not exist)
            filename,                             # for URL fallback
            clean_caption(ann['caption']),
        ))

    _, test_pairs = train_test_split(pairs, test_size=0.15, random_state=42)

    n_eval = min(n_eval, len(test_pairs))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(test_pairs), size=n_eval, replace=False)
    return [test_pairs[i] for i in idx]


def calculate_corpus_metrics(references, hypotheses):
    smoother = SmoothingFunction().method1
    refs_tok = [[r.split()] for r in references]
    hyps_tok = [h.split()   for h in hypotheses]

    b1 = corpus_bleu(refs_tok, hyps_tok, weights=(1, 0, 0, 0),             smoothing_function=smoother)
    b2 = corpus_bleu(refs_tok, hyps_tok, weights=(0.5, 0.5, 0, 0),         smoothing_function=smoother)
    b3 = corpus_bleu(refs_tok, hyps_tok, weights=(0.33, 0.33, 0.33, 0),    smoothing_function=smoother)
    b4 = corpus_bleu(refs_tok, hyps_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)

    meteor_scores, rouge_scores = [], []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    for r, h in zip(references, hypotheses):
        meteor_scores.append(meteor_score([r.split()], h.split()))
        rouge_scores.append(scorer.score(r, h)['rougeL'].fmeasure)

    return b1, b2, b3, b4, np.mean(meteor_scores), np.mean(rouge_scores)


def main():
    config  = load_config()
    vit_cfg = config.get('vit_gpt2', {})
    max_len = vit_cfg.get('max_length',   16)
    n_beams = vit_cfg.get('num_beams',     4)
    temp    = vit_cfg.get('temperature', 1.0)
    top_k   = vit_cfg.get('top_k',       50)
    top_p   = vit_cfg.get('top_p',       0.9)
    n_eval  = vit_cfg.get('eval_samples', 300)
    model_id = vit_cfg.get('model_id', 'nlpconnect/vit-gpt2-image-captioning')

    experiment_name = f"ENC_ViT_DEC_GPT2_pretrained_S{n_eval}_beams{n_beams}"
    print(f"Experiment: {experiment_name}")

    dagshub.init(
        repo_owner=config['mlflow']['repo_owner'],
        repo_name=config['mlflow']['repo_name'],
        mlflow=True,
    )
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(experiment_name)

    print("Building test pairs from COCO annotations...")
    test_pairs = build_test_pairs(config, n_eval)
    print(f"Evaluating on {len(test_pairs)} samples.")

    local_count = sum(1 for p, _, _ in test_pairs if os.path.exists(p))
    url_count   = len(test_pairs) - local_count
    print(f"  {local_count} from local disk  |  {url_count} will be fetched via COCO URL")

    print(f"Loading {model_id} ...")
    captioner = ViTGPT2Captioner()

    os.makedirs("results", exist_ok=True)
    references, hypotheses = [], []

    with mlflow.start_run(run_name="Scientific_Test_Evaluation"):
        mlflow.log_params({
            "model":        model_id,
            "max_length":   max_len,
            "num_beams":    n_beams,
            "temperature":  temp,
            "top_k":        top_k,
            "top_p":        top_p,
            "eval_samples": len(test_pairs),
        })

        for i, (img_path, filename, ref_cap) in enumerate(test_pairs):
            try:
                image = load_image(img_path, filename)
            except Exception as e:
                print(f"  ⚠️  Skipping {filename}: {e}")
                continue

            pred = captioner.generate_caption(image, max_len, n_beams, temp, top_k, top_p)
            references.append(ref_cap)
            hypotheses.append(pred)

            if i % 50 == 0:
                print(f"  [{i}/{len(test_pairs)}]  pred: {pred}")

        if not hypotheses:
            print("❌ No captions generated — check image paths and network access.")
            return

        b1, b2, b3, b4, m, r = calculate_corpus_metrics(references, hypotheses)
        metrics_dict = {
            "test_bleu1": b1, "test_bleu2": b2, "test_bleu3": b3, "test_bleu4": b4,
            "test_meteor": m, "test_rougeL": r,
        }
        mlflow.log_metrics(metrics_dict)
        print(f"\n📊 FINAL RESULTS ({len(hypotheses)} samples):\n{json.dumps(metrics_dict, indent=2)}")

        summary_path = f"results/{experiment_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        mlflow.log_artifact(summary_path)


if __name__ == "__main__":
    main()

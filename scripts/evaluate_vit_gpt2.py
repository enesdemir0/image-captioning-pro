import os
import json
import numpy as np
import mlflow
import dagshub
from PIL import Image
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
import nltk

from src.utils.config_loader import load_config
from src.data.dataset_loader import DataLoader
from src.models.vit_gpt2 import ViTGPT2Captioner

try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)


def calculate_corpus_metrics(references, hypotheses):
    smoother  = SmoothingFunction().method1
    refs_tok  = [[r.split()] for r in references]
    hyps_tok  = [h.split()   for h in hypotheses]

    b1 = corpus_bleu(refs_tok, hyps_tok, weights=(1, 0, 0, 0),                smoothing_function=smoother)
    b2 = corpus_bleu(refs_tok, hyps_tok, weights=(0.5, 0.5, 0, 0),            smoothing_function=smoother)
    b3 = corpus_bleu(refs_tok, hyps_tok, weights=(0.33, 0.33, 0.33, 0),       smoothing_function=smoother)
    b4 = corpus_bleu(refs_tok, hyps_tok, weights=(0.25, 0.25, 0.25, 0.25),    smoothing_function=smoother)

    meteor_scores, rouge_scores = [], []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    for r, h in zip(references, hypotheses):
        meteor_scores.append(meteor_score([r.split()], h.split()))
        rouge_scores.append(scorer.score(r, h)['rougeL'].fmeasure)

    return b1, b2, b3, b4, np.mean(meteor_scores), np.mean(rouge_scores)


def main():
    config   = load_config()
    vit_cfg  = config.get('vit_gpt2', {})
    max_len  = vit_cfg.get('max_length',   16)
    n_beams  = vit_cfg.get('num_beams',     4)
    temp     = vit_cfg.get('temperature', 1.0)
    top_k    = vit_cfg.get('top_k',        50)
    top_p    = vit_cfg.get('top_p',        0.9)
    n_eval   = vit_cfg.get('eval_samples', 300)
    model_id = vit_cfg.get('model_id', 'nlpconnect/vit-gpt2-image-captioning')

    experiment_name = f"ENC_ViT_DEC_GPT2_pretrained_S{n_eval}_beams{n_beams}"
    print(f"Experiment: {experiment_name}")

    dagshub.init(
        repo_owner=config['mlflow']['repo_owner'],
        repo_name=config['mlflow']['repo_name'],
        mlflow=True
    )
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(experiment_name)

    # ── Load COCO test split ─────────────────────────────────────────────────
    loader = DataLoader(config)
    img_paths, captions = loader.load_annotations()
    _, _, (test_imgs, test_caps) = loader.split_data(img_paths, captions)

    total     = len(test_imgs)
    n_eval    = min(n_eval, total)
    rng       = np.random.default_rng(42)
    indices   = rng.choice(total, size=n_eval, replace=False)
    test_imgs = [test_imgs[i] for i in indices]
    test_caps = [test_caps[i] for i in indices]
    print(f"Evaluating on {n_eval} random test samples.")

    # ── Load ViT-GPT2 ────────────────────────────────────────────────────────
    print(f"Loading {model_id} ...")
    captioner = ViTGPT2Captioner()

    # ── Generate captions ────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)
    references, hypotheses = [], []

    with mlflow.start_run(run_name="ViTGPT2_Test_Evaluation"):
        mlflow.log_params({
            "model":       model_id,
            "max_length":  max_len,
            "num_beams":   n_beams,
            "temperature": temp,
            "top_k":       top_k,
            "top_p":       top_p,
            "eval_samples": n_eval,
        })

        for i, (img_path, ref_cap) in enumerate(zip(test_imgs, test_caps)):
            image     = Image.open(img_path).convert("RGB")
            pred      = captioner.generate_caption(image, max_len, n_beams, temp, top_k, top_p)
            ref_clean = ref_cap.replace('<start>', '').replace('<end>', '').strip()

            references.append(ref_clean)
            hypotheses.append(pred)

            if i % 50 == 0:
                print(f"  Processed {i}/{n_eval}  |  last pred: {pred}")

        b1, b2, b3, b4, m, r = calculate_corpus_metrics(references, hypotheses)
        metrics = {"BLEU-1": b1, "BLEU-2": b2, "BLEU-3": b3, "BLEU-4": b4,
                   "METEOR": m, "ROUGE-L": r}

        mlflow.log_metrics(metrics)
        print(f"\nFINAL RESULTS ({n_eval} SAMPLES):\n{json.dumps(metrics, indent=2)}")

        summary_path = f"results/{experiment_name}_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        mlflow.log_artifact(summary_path)


if __name__ == "__main__":
    main()

import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn


def calculate_metrics(references, hypotheses):
    """
    Identical metric set to main branch plus BERTScore-F1.
    Keys are prefixed with test_ to match MLflow convention.
    """
    smoother = SmoothingFunction().method1
    refs_tok = [[r.split()] for r in references]
    hyps_tok = [h.split() for h in hypotheses]

    b1 = corpus_bleu(refs_tok, hyps_tok, weights=(1, 0, 0, 0), smoothing_function=smoother)
    b2 = corpus_bleu(refs_tok, hyps_tok, weights=(0.5, 0.5, 0, 0), smoothing_function=smoother)
    b3 = corpus_bleu(refs_tok, hyps_tok, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoother)
    b4 = corpus_bleu(refs_tok, hyps_tok, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoother)

    meteor_scores, rouge_scores = [], []
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    for r, h in zip(references, hypotheses):
        meteor_scores.append(meteor_score([r.split()], h.split()))
        rouge_scores.append(scorer.score(r, h)['rougeL'].fmeasure)

    _, _, F1 = bert_score_fn(hypotheses, references, lang="en", verbose=False)

    return {
        "test_bleu1":        float(b1),
        "test_bleu2":        float(b2),
        "test_bleu3":        float(b3),
        "test_bleu4":        float(b4),
        "test_meteor":       float(np.mean(meteor_scores)),
        "test_rougeL":       float(np.mean(rouge_scores)),
        "test_bertscore_f1": float(F1.mean()),
    }

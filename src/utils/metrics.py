from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer

def calculate_metrics(real_caption, predicted_caption):
    """
    Calculates BLEU-4 and ROUGE-L for a single pair of captions.
    """
    # Clean the captions (remove <start> and <end>)
    real = real_caption.replace('<start>', '').replace('<end>', '').strip().split()
    pred = predicted_caption.replace('<start>', '').replace('<end>', '').strip().split()

    # 1. BLEU Score (with smoothing to avoid 0.0 for short sentences)
    smooth = SmoothingFunction().method1
    # We use [real] because BLEU expects a list of references
    bleu4 = sentence_bleu([real], pred, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    # 2. ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(' '.join(real), ' '.join(pred))
    rouge_l = scores['rougeL'].fmeasure

    return bleu4, rouge_l
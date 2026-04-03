import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# Ensure NLTK resources are available
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt', quiet=True)

def calculate_all_metrics(real_caption, predicted_caption):
    """Calculates BLEU-4, METEOR, and ROUGE-L."""
    real_words = real_caption.replace('<start>', '').replace('<end>', '').strip().split()
    pred_words = predicted_caption.replace('<start>', '').replace('<end>', '').strip().split()

    # 1. BLEU-4
    smooth = SmoothingFunction().method1
    bleu4 = sentence_bleu([real_words], pred_words, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    # 2. ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(' '.join(real_words), ' '.join(pred_words))
    rouge_l = scores['rougeL'].fmeasure

    # 3. METEOR
    meteor = meteor_score([real_words], pred_words)

    return bleu4, meteor, rouge_l
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

# Ensure all NLTK resources are ready
try:
    nltk.data.find('corpora/wordnet')
    nltk.data.find('corpora/omw-1.4')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('wordnet', quiet=True)
    nltk.download('omw-1.4', quiet=True)
    nltk.download('punkt', quiet=True)

def calculate_all_metrics(real_caption, predicted_caption):
    """
    Calculates BLEU 1-4, METEOR, and ROUGE-L.
    Returns: (bleu_tuple, meteor, rouge_l)
    """
    # Clean and tokenize
    real_words = real_caption.replace('<start>', '').replace('<end>', '').strip().split()
    pred_words = predicted_caption.replace('<start>', '').replace('<end>', '').strip().split()

    # Smoothing function is essential for short sentences
    smooth = SmoothingFunction().method1
    
    # 1. BLEU Scores with specific weights
    b1 = sentence_bleu([real_words], pred_words, weights=(1, 0, 0, 0), smoothing_function=smooth)
    b2 = sentence_bleu([real_words], pred_words, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)
    b3 = sentence_bleu([real_words], pred_words, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth)
    b4 = sentence_bleu([real_words], pred_words, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)

    # 2. ROUGE-L
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = scorer.score(' '.join(real_words), ' '.join(pred_words))['rougeL'].fmeasure

    # 3. METEOR
    meteor = meteor_score([real_words], pred_words)

    return (b1, b2, b3, b4), meteor, rouge_l
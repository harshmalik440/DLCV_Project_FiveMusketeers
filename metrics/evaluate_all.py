"""
Comprehensive evaluation script computing:
- BLEU (1-4)
- ROUGE-L
- Pseudo-Gloss Entity Recall (PGER)
- Translation Soft Recall (TSR)
"""

import numpy as np
from .translation_metrics import compute_bleu_and_rouge
from .semantic_fidelity import compute_pger, compute_tsr

def evaluate_predictions(predictions, references, lang="de"):
    """
    Args:
        predictions: list of predicted strings
        references: list of reference strings
        lang: language code ('de' for German PHOENIX-2014T, 'en' for ASL/How2Sign)
    Returns:
        dict containing all metric scores
    """
    assert len(predictions) == len(references), "Count mismatch between predictions and references"

    # Compute BLEU and ROUGE-L
    res = compute_bleu_and_rouge(predictions, references)

    # Compute PGER and TSR per sample
    pger_scores = []
    tsr_scores = []
    for pred, ref in zip(predictions, references):
        pger_scores.append(compute_pger(pred, ref, lang=lang))
        tsr_scores.append(compute_tsr(pred, ref))

    res["PGER"] = round(float(np.mean(pger_scores)), 4)
    res["TSR"] = round(float(np.mean(tsr_scores)), 4)

    return res

def print_evaluation_summary(metrics_dict, model_name="Model"):
    print(f"\n================ Evaluation Results: {model_name} ================")
    print(f"  BLEU-1  : {metrics_dict.get('BLEU1', 0.0):.2f}")
    print(f"  BLEU-2  : {metrics_dict.get('BLEU2', 0.0):.2f}")
    print(f"  BLEU-3  : {metrics_dict.get('BLEU3', 0.0):.2f}")
    print(f"  BLEU-4  : {metrics_dict.get('BLEU4', 0.0):.2f}")
    print(f"  ROUGE-L : {metrics_dict.get('ROUGE-L', 0.0):.2f}")
    print(f"  PGER    : {metrics_dict.get('PGER', 0.0):.4f}")
    print(f"  TSR     : {metrics_dict.get('TSR', 0.0):.4f}")
    print("==================================================================\n")


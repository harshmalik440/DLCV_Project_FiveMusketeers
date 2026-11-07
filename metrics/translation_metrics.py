"""
Standard Machine Translation Metrics:
- BLEU (BLEU-1, BLEU-2, BLEU-3, BLEU-4) using SacreBLEU format
- ROUGE-L (Longest Common Subsequence)
"""

import collections
import math

try:
    import sacrebleu
    HAS_SACREBLEU = True
except ImportError:
    HAS_SACREBLEU = False

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE = True
except ImportError:
    HAS_ROUGE = False

def _ngram_counts(tokens, n):
    if len(tokens) < n:
        return collections.Counter()
    return collections.Counter([tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)])

def _pure_python_bleu(hypotheses, references, max_n=4):
    """
    Standard corpus-level BLEU computation fallback when sacrebleu is not installed.
    Matches standard SacreBLEU precision clipping and brevity penalty.
    """
    precisions = [0.0] * max_n
    total_hyp_len = 0
    total_ref_len = 0

    clipped_counts = [0] * max_n
    total_counts = [0] * max_n

    for hyp, ref in zip(hypotheses, references):
        hyp_toks = hyp.strip().split()
        ref_toks = ref.strip().split()
        total_hyp_len += len(hyp_toks)
        total_ref_len += len(ref_toks)

        for n in range(1, max_n + 1):
            hyp_ngrams = _ngram_counts(hyp_toks, n)
            ref_ngrams = _ngram_counts(ref_toks, n)

            clipped = 0
            for ng, count in hyp_ngrams.items():
                clipped += min(count, ref_ngrams.get(ng, 0))

            clipped_counts[n - 1] += clipped
            total_counts[n - 1] += max(0, len(hyp_toks) - n + 1)

    for n in range(max_n):
        if total_counts[n] > 0 and clipped_counts[n] > 0:
            precisions[n] = clipped_counts[n] / total_counts[n]
        else:
            precisions[n] = 1e-9

    # Brevity penalty
    if total_hyp_len == 0:
        return [0.0] * max_n

    if total_hyp_len > total_ref_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - total_ref_len / max(1, total_hyp_len))

    bleu_scores = []
    log_sum = 0.0
    for n in range(max_n):
        log_sum += math.log(precisions[n])
        bleu_n = bp * math.exp(log_sum / (n + 1)) * 100.0
        bleu_scores.append(round(bleu_n, 2))

    return bleu_scores

def _pure_python_rouge_l(hypotheses, references):
    """
    Pure Python ROUGE-L fallback calculating Longest Common Subsequence F1.
    """
    f1_scores = []
    for hyp, ref in zip(hypotheses, references):
        h_toks = hyp.strip().split()
        r_toks = ref.strip().split()
        m, n = len(h_toks), len(r_toks)
        if m == 0 or n == 0:
            f1_scores.append(0.0)
            continue

        # DP table for LCS
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if h_toks[i - 1] == r_toks[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs = dp[m][n]
        prec = lcs / m
        rec = lcs / n
        if prec + rec > 0:
            f1 = 2 * (prec * rec) / (prec + rec)
        else:
            f1 = 0.0
        f1_scores.append(f1 * 100.0)

    return round(sum(f1_scores) / max(1, len(f1_scores)), 2)

def compute_bleu_and_rouge(predictions, references):
    """
    Computes BLEU-1, BLEU-2, BLEU-3, BLEU-4 and ROUGE-L for predictions against references.
    """
    res = {}
    if HAS_SACREBLEU:
        try:
            # References format for sacrebleu is list of reference streams: [[ref1, ref2, ...]]
            refs_wrapped = [references]
            bleu_1 = sacrebleu.corpus_bleu(predictions, refs_wrapped, max_ngram_order=1).score
            bleu_2 = sacrebleu.corpus_bleu(predictions, refs_wrapped, max_ngram_order=2).score
            bleu_3 = sacrebleu.corpus_bleu(predictions, refs_wrapped, max_ngram_order=3).score
            bleu_4 = sacrebleu.corpus_bleu(predictions, refs_wrapped, max_ngram_order=4).score
            res["BLEU1"] = round(bleu_1, 2)
            res["BLEU2"] = round(bleu_2, 2)
            res["BLEU3"] = round(bleu_3, 2)
            res["BLEU4"] = round(bleu_4, 2)
        except Exception:
            py_bleu = _pure_python_bleu(predictions, references)
            res["BLEU1"], res["BLEU2"], res["BLEU3"], res["BLEU4"] = py_bleu
    else:
        py_bleu = _pure_python_bleu(predictions, references)
        res["BLEU1"], res["BLEU2"], res["BLEU3"], res["BLEU4"] = py_bleu

    if HAS_ROUGE:
        try:
            scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
            scores = [scorer.score(ref, pred)['rougeL'].fmeasure * 100.0
                      for pred, ref in zip(predictions, references)]
            res["ROUGE-L"] = round(sum(scores) / max(1, len(scores)), 2)
        except Exception:
            res["ROUGE-L"] = _pure_python_rouge_l(predictions, references)
    else:
        res["ROUGE-L"] = _pure_python_rouge_l(predictions, references)

    return res


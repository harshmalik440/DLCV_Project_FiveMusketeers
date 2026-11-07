"""
Unit tests for translation and semantic fidelity evaluation metrics:
- BLEU-1 to BLEU-4
- ROUGE-L
- Pseudo-Gloss Entity Recall (PGER)
- Translation Soft Recall (TSR)
"""

import unittest
from metrics.translation_metrics import compute_bleu_and_rouge, _pure_python_bleu, _pure_python_rouge_l
from metrics.semantic_fidelity import compute_pger, compute_tsr, extract_content_entities

class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.references = [
            "und nun die wettervorhersage für morgen freitag den fünfzehnten oktober",
            "am freitag ist es in der nähe der küsten und ganz im süden teilweise stark bewölkt"
        ]
        self.perfect_preds = [
            "und nun die wettervorhersage für morgen freitag den fünfzehnten oktober",
            "am freitag ist es in der nähe der küsten und ganz im süden teilweise stark bewölkt"
        ]
        self.perturbed_preds = [
            "und nun die wettervorhersage für morgen freitag den dreißigsten mai",
            "am freitag ist es im süden und osten weiter freundlich im norden regnet es gebietsweise"
        ]

    def test_perfect_matches(self):
        res = compute_bleu_and_rouge(self.perfect_preds, self.references)
        self.assertAlmostEqual(res["BLEU1"], 100.0, places=1)
        self.assertAlmostEqual(res["BLEU4"], 100.0, places=1)
        self.assertAlmostEqual(res["ROUGE-L"], 100.0, places=1)

    def test_perturbed_matches(self):
        res = compute_bleu_and_rouge(self.perturbed_preds, self.references)
        self.assertTrue(0.0 <= res["BLEU4"] < 100.0)
        self.assertTrue(0.0 <= res["ROUGE-L"] < 100.0)

    def test_pger_extraction(self):
        ref = "am freitag ist es im norden bewölkt mit regen"
        pred = "am freitag regnet es im norden"
        pger = compute_pger(pred, ref, lang="de")
        self.assertTrue(0.0 < pger <= 1.0)

    def test_tsr_metric(self):
        ref = "wettervorhersage für morgen freitag den fünfzehnten oktober"
        pred = "wetterbericht für freitag den 15 oktober"
        tsr = compute_tsr(pred, ref)
        self.assertTrue(0.0 <= tsr <= 1.0)

if __name__ == "__main__":
    unittest.main()


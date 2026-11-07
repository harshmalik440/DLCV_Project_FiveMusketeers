"""
Quick standalone verification of pure Python modules:
- SimpleSignTokenizer
- Metrics (Pure Python BLEU, Pure Python ROUGE-L, PGER, TSR)
- Sample dataset generation
"""

import sys
import os

print("--- Running Standalone Verification ---")

# Test 1: Sample Data Generation
from data.sample_data import generate_mock_phoenix_dataset, SAMPLE_SENTENCES
mock_dir = "./data/raw/phoenix2014t"
generate_mock_phoenix_dataset(mock_dir, num_dummy_frames=4)
assert os.path.exists(os.path.join(mock_dir, "annotations", "train.json")), "Mock train.json missing"
print("[Test 1 PASSED] Mock data generation successful.")

# Test 2: Tokenizer
from data.tokenizer import SimpleSignTokenizer
tokenizer = SimpleSignTokenizer()
texts = [s["text"] for s in SAMPLE_SENTENCES]
tokenizer.build_vocab(texts)
sample_text = "und nun die wettervorhersage"
encoded = tokenizer.encode(sample_text, max_len=8)
decoded = tokenizer.decode(encoded)
assert "wettervorhersage" in decoded, "Tokenizer encoding/decoding failed"
print(f"[Test 2 PASSED] Tokenizer vocab: {tokenizer.vocab_size} tokens, Decoded: '{decoded}'")

# Test 3: Metrics
from metrics.translation_metrics import _pure_python_bleu, _pure_python_rouge_l
from metrics.semantic_fidelity import compute_pger, compute_tsr

refs = ["und nun die wettervorhersage für morgen freitag den fünfzehnten oktober"]
preds = ["und nun die wettervorhersage für morgen freitag den fünfzehnten oktober"]

bleus = _pure_python_bleu(preds, refs)
rouge_l = _pure_python_rouge_l(preds, refs)
pger = compute_pger(preds[0], refs[0], lang="de")
tsr = compute_tsr(preds[0], refs[0])

print(f"[Test 3 PASSED] Metrics on identical sentence:")
print(f"  BLEU (1-4): {bleus}")
print(f"  ROUGE-L   : {rouge_l}")
print(f"  PGER      : {pger}")
print(f"  TSR       : {tsr}")

assert bleus[0] == 100.0, "BLEU-1 should be 100.0 for identical text"
assert rouge_l == 100.0, "ROUGE-L should be 100.0 for identical text"
assert pger == 1.0, "PGER should be 1.0 for identical text"
assert abs(tsr - 1.0) < 1e-4, "TSR should be approx 1.0 for identical text"

print("\n>>> ALL STANDALONE VERIFICATION TESTS PASSED SUCCESSFULLY! <<<")


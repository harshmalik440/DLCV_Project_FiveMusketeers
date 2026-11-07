"""
Evaluation script for Baseline and Proposed DynST-VLM models.
Computes:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4
- ROUGE-L
- Pseudo-Gloss Entity Recall (PGER)
- Translation Soft Recall (TSR)
"""

import os
import argparse
import json
import torch
from torch.utils.data import DataLoader

from data.dataset import SignLanguageDataset, collate_slt_batch
from data.sample_data import generate_mock_phoenix_dataset
from models.vlm_baseline import BaselineVLMForSLT
from models.vlm_proposed import DynSTVLMForSLT
from metrics.evaluate_all import evaluate_predictions, print_evaluation_summary
from data.tokenizer import SimpleSignTokenizer

def run_evaluation(model, dataloader, tokenizer, device, max_len=32):
    model.eval()
    predictions = []
    references = []

    with torch.no_grad():
        for batch in dataloader:
            frames = batch["frames"].to(device)
            targets = batch["target_texts"]

            bos_id = tokenizer.token2id.get(tokenizer.bos_token, 1)
            eos_id = tokenizer.token2id.get(tokenizer.eos_token, 2)

            generated_ids = model.generate(frames, max_len=max_len, bos_id=bos_id, eos_id=eos_id)

            for i in range(generated_ids.size(0)):
                pred_str = tokenizer.decode(generated_ids[i].cpu().tolist())
                predictions.append(pred_str)
                references.append(targets[i])

    return predictions, references

def main():
    parser = argparse.ArgumentParser(description="Evaluate Sign Language Translation Model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt file)")
    parser.add_argument("--data_dir", type=str, default="./data/raw/phoenix2014t")
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test", "train"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    if not os.path.exists(os.path.join(args.data_dir, "annotations", f"{args.split}.json")):
        generate_mock_phoenix_dataset(args.data_dir)

    print(f"\nLoading checkpoint from: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model_type = ckpt.get("model_type", "proposed")

    tokenizer = SimpleSignTokenizer()
    tokenizer.token2id = ckpt["vocab"]
    tokenizer.id2token = {v: k for k, v in ckpt["vocab"].items()}

    test_dataset = SignLanguageDataset(data_dir=args.data_dir, split=args.split, max_frames=64)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_slt_batch)

    if model_type == "proposed":
        model = DynSTVLMForSLT(
            vision_dim=1152,
            llm_dim=2048,
            vocab_size=max(tokenizer.vocab_size + 10, 100),
            num_articulator_queries=16,
            contrastive_dim=512
        )
    else:
        model = BaselineVLMForSLT(
            vision_dim=1152,
            llm_dim=2048,
            vocab_size=max(tokenizer.vocab_size + 10, 100)
        )

    model.load_state_dict(ckpt["model_state_dict"])
    model.to(args.device)

    print(f"Running inference on {len(test_dataset)} samples from '{args.split}' split...")
    preds, refs = run_evaluation(model, test_loader, tokenizer, args.device)

    metrics = evaluate_predictions(preds, refs, lang="de")
    print_evaluation_summary(metrics, model_name=f"{model_type.upper()} on {args.split.upper()}")

    print("Sample Qualitative Predictions:")
    for i in range(min(3, len(preds))):
        print(f"\n[Sample {i+1}]")
        print(f"  Reference : {refs[i]}")
        print(f"  Predicted : {preds[i]}")

    if args.output_file:
        out_data = {
            "metrics": metrics,
            "samples": [{"reference": r, "predicted": p} for r, p in zip(refs, preds)]
        }
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {args.output_file}")

if __name__ == "__main__":
    main()


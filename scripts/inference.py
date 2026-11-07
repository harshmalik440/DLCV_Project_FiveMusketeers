"""
Single sample inference and visual demo script.
Given a video or directory of frames, predicts the translated spoken text.
"""

import os
import argparse
import torch
import torchvision.transforms as T
from PIL import Image

from data.dataset import SignLanguageDataset, DGS_PROMPT
from models.vlm_proposed import DynSTVLMForSLT
from models.vlm_baseline import BaselineVLMForSLT
from data.tokenizer import SimpleSignTokenizer

def main():
    parser = argparse.ArgumentParser(description="Run inference on a single sign language sequence")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--sample_id", type=str, default="phoenix_test_001", help="Sample ID in data directory")
    parser.add_argument("--data_dir", type=str, default="./data/raw/phoenix2014t")
    parser.add_argument("--max_frames", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model_type = ckpt.get("model_type", "proposed")

    tokenizer = SimpleSignTokenizer()
    tokenizer.token2id = ckpt["vocab"]
    tokenizer.id2token = {v: k for k, v in ckpt["vocab"].items()}

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
    model.eval()

    # Load sample frames
    dataset = SignLanguageDataset(data_dir=args.data_dir, split="test", max_frames=args.max_frames)
    target_sample = None
    for s in dataset.samples:
        if s["id"] == args.sample_id:
            target_sample = s
            break

    if target_sample is None:
        target_sample = dataset.samples[0]
        args.sample_id = target_sample["id"]

    frames = dataset.load_frames(args.sample_id).unsqueeze(0).to(args.device)

    print(f"\nTarget Sample ID : {args.sample_id}")
    print(f"Ground Truth Ref : {target_sample.get('text', 'N/A')}")
    print(f"Sign Gloss Ref   : {target_sample.get('gloss', 'N/A')}")

    bos_id = tokenizer.token2id.get(tokenizer.bos_token, 1)
    eos_id = tokenizer.token2id.get(tokenizer.eos_token, 2)

    with torch.no_grad():
        out_ids = model.generate(frames, max_len=32, bos_id=bos_id, eos_id=eos_id)
        pred_text = tokenizer.decode(out_ids[0].cpu().tolist())

    print(f"\n================ Prediction ================")
    print(f"Model ({model_type.upper()}): {pred_text}")
    print(f"============================================")

if __name__ == "__main__":
    main()


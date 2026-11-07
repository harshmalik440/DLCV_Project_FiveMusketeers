"""
Training pipeline for Sign Language Translation.
Supports both Baseline VLM and Proposed DynST-VLM.

Fixes applied (audit items G1-G6):
  G2: Validation-based checkpoint selection (dev set eval every epoch).
  G3: Cosine / constant LR scheduler from config.
  G4: Padding tokens excluded from CE loss via ignore_index=pad_id.
  G5: Vocabulary built from training set only (dev set excluded).
  G6: img_size read from config and passed to dataset & model constructors.
"""

import os
import math
import argparse
import yaml
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.dataset import SignLanguageDataset, collate_slt_batch
from data.sample_data import generate_mock_phoenix_dataset
from data.tokenizer import SimpleSignTokenizer
from models.vlm_baseline import BaselineVLMForSLT
from models.vlm_proposed import DynSTVLMForSLT
from models.losses import DynSTLoss


def train_epoch(model, dataloader, optimizer, criterion, tokenizer, device,
                grad_accum_steps=4, is_proposed=False):
    model.train()
    total_loss = 0.0
    num_steps = 0
    optimizer.zero_grad()

    pad_id = tokenizer.token2id.get(tokenizer.pad_token, 0)

    for step, batch in enumerate(dataloader):
        frames = batch["frames"].to(device)
        targets = batch["target_texts"]

        token_ids_list = [tokenizer.encode(t) for t in targets]
        text_ids = torch.tensor(token_ids_list, dtype=torch.long, device=device)

        inputs = text_ids[:, :-1]
        labels = text_ids[:, 1:].clone()
        # Mask padding tokens so they don't contribute to loss (Fix G4)
        labels[labels == pad_id] = criterion.ce_loss.ignore_index

        if is_proposed:
            output = model(frames, text_ids=inputs)
            logits = output["logits"]
            video_proj = output["video_proj"]
            text_proj = output["text_proj"]
            loss, loss_dict = criterion(logits, labels,
                                        video_repr=video_proj, text_repr=text_proj)
        else:
            output = model(frames, text_ids=inputs)
            logits = output["logits"]
            loss, loss_dict = criterion(logits, labels)

        loss_scaled = loss / grad_accum_steps
        loss_scaled.backward()

        total_loss += loss.item()
        num_steps += 1

        if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(dataloader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

    return total_loss / max(1, num_steps)


@torch.no_grad()
def evaluate_dev(model, dataloader, criterion, tokenizer, device, is_proposed=False):
    """Compute average loss on the dev set for checkpoint selection (Fix G2)."""
    model.eval()
    total_loss = 0.0
    num_steps = 0
    pad_id = tokenizer.token2id.get(tokenizer.pad_token, 0)

    for batch in dataloader:
        frames = batch["frames"].to(device)
        targets = batch["target_texts"]

        token_ids_list = [tokenizer.encode(t) for t in targets]
        text_ids = torch.tensor(token_ids_list, dtype=torch.long, device=device)

        inputs = text_ids[:, :-1]
        labels = text_ids[:, 1:].clone()
        labels[labels == pad_id] = criterion.ce_loss.ignore_index

        if is_proposed:
            output = model(frames, text_ids=inputs)
            loss, _ = criterion(output["logits"], labels,
                                video_repr=output["video_proj"],
                                text_repr=output["text_proj"])
        else:
            output = model(frames, text_ids=inputs)
            loss, _ = criterion(output["logits"], labels)

        total_loss += loss.item()
        num_steps += 1

    return total_loss / max(1, num_steps)


def main():
    parser = argparse.ArgumentParser(description="Train Baseline or Proposed DynST-VLM for SLT")
    parser.add_argument("--config", type=str, default="configs/proposed_dynst_config.yaml")
    parser.add_argument("--model_type", type=str, choices=["baseline", "proposed"], default="proposed")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pretrained", action="store_true",
                        help="Use real HuggingFace pretrained VLM (requires transformers + peft).")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_dir = cfg["dataset"]["data_dir"]
    img_size = cfg["dataset"].get("img_size", 224)  # Fix G6: read from config

    if not os.path.exists(os.path.join(data_dir, "annotations", "train.json")):
        print(f"[Setup] Dataset not found in {data_dir}. Generating sample dataset...")
        generate_mock_phoenix_dataset(data_dir)

    print(f"\n=======================================================")
    print(f"  Training Mode : {args.model_type.upper()}")
    print(f"  Pretrained    : {args.pretrained}")
    print(f"  Image Size    : {img_size}")
    print(f"  Configuration : {args.config}")
    print(f"  Device        : {args.device}")
    print(f"=======================================================\n")

    # Fix G6: pass img_size from config to dataset
    train_dataset = SignLanguageDataset(data_dir=data_dir, split="train",
                                        max_frames=cfg["dataset"]["max_frames"],
                                        img_size=img_size)
    dev_dataset = SignLanguageDataset(data_dir=data_dir, split="dev",
                                      max_frames=cfg["dataset"]["max_frames"],
                                      img_size=img_size)

    # Fix G5: Build vocabulary from training set ONLY (not dev)
    tokenizer = SimpleSignTokenizer()
    train_texts = [s["text"] for s in train_dataset.samples]
    tokenizer.build_vocab(train_texts)
    print(f"[Tokenizer] Vocabulary size: {tokenizer.vocab_size} tokens (train-only)")

    pad_id = tokenizer.token2id.get(tokenizer.pad_token, 0)

    train_loader = DataLoader(train_dataset, batch_size=cfg["training"]["batch_size"],
                              shuffle=True, collate_fn=collate_slt_batch)
    dev_loader = DataLoader(dev_dataset, batch_size=cfg["training"]["batch_size"],
                            shuffle=False, collate_fn=collate_slt_batch)

    is_proposed = (args.model_type == "proposed")
    vocab_size = max(tokenizer.vocab_size + 10, 100)

    if is_proposed:
        model = DynSTVLMForSLT(
            vision_dim=cfg["model"]["vision_dim"],
            llm_dim=cfg["model"]["llm_dim"],
            vocab_size=vocab_size,
            img_size=img_size,  # Fix G6
            num_articulator_queries=cfg["model"]["motion_stream"]["gating_reduction"] * 4,
            contrastive_dim=cfg["model"]["contrastive_head"]["projection_dim"],
            lora_r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["lora_alpha"],
            pretrained=args.pretrained
        )
        c_weight = cfg["model"]["contrastive_head"]["loss_weight"]
        c_tau = cfg["model"]["contrastive_head"]["temperature"]
        c_dim = cfg["model"]["contrastive_head"]["projection_dim"]
        criterion = DynSTLoss(contrastive_weight=c_weight, temperature=c_tau,
                              ignore_index=pad_id, contrastive_dim=c_dim)
    else:
        model = BaselineVLMForSLT(
            vision_dim=cfg["model"]["vision_dim"],
            llm_dim=cfg["model"]["llm_dim"],
            vocab_size=vocab_size,
            img_size=img_size,  # Fix G6
            lora_r=cfg["lora"]["r"],
            lora_alpha=cfg["lora"]["lora_alpha"],
            pretrained=args.pretrained
        )
        criterion = DynSTLoss(contrastive_weight=0.0, ignore_index=pad_id)

    model.to(args.device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    total_trainable = sum(p.numel() for p in trainable_params)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[Model] Trainable: {total_trainable:,} / Total: {total_params:,} "
          f"({total_trainable / total_params * 100:.2f}%)")

    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"])
    )

    # Fix G3: Implement LR scheduler from config
    epochs = args.epochs or cfg["training"]["epochs"]
    scheduler_type = cfg["training"].get("lr_scheduler", "constant")
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
        print(f"[Scheduler] Cosine annealing over {epochs} epochs")
    else:
        scheduler = None
        print(f"[Scheduler] Constant learning rate")

    ckpt_dir = cfg["training"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)

    print(f"\nStarting training for {epochs} epochs...")
    best_dev_loss = float("inf")
    ckpt_path = os.path.join(ckpt_dir, f"{args.model_type}_best.pt")

    for ep in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion,
                                 tokenizer, args.device,
                                 grad_accum_steps=cfg["training"]["gradient_accumulation_steps"],
                                 is_proposed=is_proposed)

        # Fix G2: Select checkpoint based on dev loss, not training loss
        dev_loss = evaluate_dev(model, dev_loader, criterion, tokenizer,
                                args.device, is_proposed=is_proposed)

        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Epoch [{ep:02d}/{epochs:02d}] - Train Loss: {train_loss:.4f} | "
              f"Dev Loss: {dev_loss:.4f} | LR: {lr_now:.2e}")

        if scheduler is not None:
            scheduler.step()

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "vocab": tokenizer.token2id,
                "train_loss": train_loss,
                "dev_loss": dev_loss,
                "model_type": args.model_type,
                "img_size": img_size
            }, ckpt_path)
            print(f"    -> New best dev loss! Checkpoint saved.")

    print(f"\n[Training Completed] Best checkpoint (dev loss={best_dev_loss:.4f}) at: {ckpt_path}")


if __name__ == "__main__":
    main()

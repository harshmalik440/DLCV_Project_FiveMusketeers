"""
Sign Language Translation Dataset implementation.
Implements the exact deterministic uniform frame sampling from the paper:
Eq (1): f_i = frame[ floor( (i + 1) * T / T_max ) - 1 ]
"""

import os
import glob
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

DGS_PROMPT = "Übersetzen Sie das Video in deutscher Gebärdensprache in deutschen Text (gesprochene Sprache)."
ASL_PROMPT = "Translate the English sign language video into spoken text."
ISL_PROMPT = "Translate the Indian sign language video into spoken text."

class SignLanguageDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="train",
        max_frames=64,
        img_size=224,
        prompt=DGS_PROMPT,
        transform=None,
        feature_cache_dir=None
    ):
        """
        Args:
            data_dir: Path to dataset root containing 'annotations/' and 'frames/'
            split: 'train', 'dev', or 'test'
            max_frames: T_max as specified in paper (default: 64)
            img_size: Spatial resolution for frames (e.g. 224 or 384)
            prompt: Task instruction prompt (from Table 17 in paper)
        """
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_frames = max_frames
        self.img_size = img_size
        self.prompt = prompt
        self.feature_cache_dir = feature_cache_dir

        if transform is not None:
            self.transform = transform
        else:
            self.transform = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        ann_path = os.path.join(data_dir, "annotations", f"{split}.json")
        if not os.path.exists(ann_path):
            raise FileNotFoundError(f"Annotations not found at {ann_path}")

        with open(ann_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

    def sample_frame_indices(self, total_frames):
        """
        Deterministic uniform sampling from paper Eq. (1):
        fi = frame[ floor( (i + 1) * T / T_max ) - 1 ]
        """
        T_total = total_frames
        T_max = self.max_frames

        if T_total <= T_max:
            return list(range(T_total))

        indices = []
        for i in range(T_max):
            idx = int(np_floor := ((i + 1) * T_total) // T_max) - 1
            idx = max(0, min(idx, T_total - 1))
            indices.append(idx)
        return indices

    def load_frames(self, sample_id):
        sample_frame_dir = os.path.join(self.data_dir, "frames", sample_id)
        if not os.path.exists(sample_frame_dir):
            # Fallback if frames not extracted or in alternate folder
            # Create a zero tensor fallback
            return torch.zeros((self.max_frames, 3, self.img_size, self.img_size))

        frame_files = sorted(glob.glob(os.path.join(sample_frame_dir, "*.jpg")) +
                             glob.glob(os.path.join(sample_frame_dir, "*.png")))
        if len(frame_files) == 0:
            return torch.zeros((self.max_frames, 3, self.img_size, self.img_size))

        sampled_indices = self.sample_frame_indices(len(frame_files))
        selected_files = [frame_files[idx] for idx in sampled_indices]

        frames = []
        for fpath in selected_files:
            try:
                img = Image.open(fpath).convert("RGB")
                frames.append(self.transform(img))
            except Exception:
                frames.append(torch.zeros(3, self.img_size, self.img_size))

        frames_tensor = torch.stack(frames, dim=0) # [T, C, H, W]

        # Pad to max_frames if video is shorter than max_frames
        if frames_tensor.size(0) < self.max_frames:
            pad_count = self.max_frames - frames_tensor.size(0)
            last_frame = frames_tensor[-1:].repeat(pad_count, 1, 1, 1)
            frames_tensor = torch.cat([frames_tensor, last_frame], dim=0)

        return frames_tensor[:self.max_frames]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        sample_id = item["id"]
        target_text = item.get("text", "")
        gloss = item.get("gloss", "")

        frames = self.load_frames(sample_id)

        return {
            "id": sample_id,
            "frames": frames, # [T_max, C, H, W]
            "prompt": self.prompt,
            "target_text": target_text,
            "gloss": gloss
        }

def collate_slt_batch(batch):
    """
    Collate function for sign language translation.
    """
    ids = [b["id"] for b in batch]
    prompts = [b["prompt"] for b in batch]
    target_texts = [b["target_text"] for b in batch]
    glosses = [b["gloss"] for b in batch]
    frames = torch.stack([b["frames"] for b in batch], dim=0) # [B, T, C, H, W]

    return {
        "ids": ids,
        "frames": frames,
        "prompts": prompts,
        "target_texts": target_texts,
        "glosses": glosses
    }


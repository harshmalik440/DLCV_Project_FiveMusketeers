"""
Generates a realistic sample dataset for PHOENIX-2014T (German Sign Language),
mirroring the structure and linguistic characteristics reported in the ACL 2026 paper.
Enables immediate testing and verification without requiring the full multi-GB download.
"""

import os
import json
import numpy as np
from PIL import Image

SAMPLE_SENTENCES = [
    {
        "id": "phoenix_train_001",
        "signer": "Signer01",
        "text": "und nun die wettervorhersage für morgen freitag den fünfzehnten oktober",
        "gloss": "JETZT WETTER MORGEN FREITAG FUNFZEHN OKTOBER",
        "num_frames": 115
    },
    {
        "id": "phoenix_train_002",
        "signer": "Signer02",
        "text": "am freitag ist es in der nähe der küsten und ganz im süden teilweise stark bewölkt",
        "gloss": "FREITAG KUESTE REGION SUED TEILWEISE WOLKIG REGEN",
        "num_frames": 142
    },
    {
        "id": "phoenix_train_003",
        "signer": "Signer03",
        "text": "im osten und südosten freundlich und trocken mit viel sonnenschein",
        "gloss": "OST SUEDOST FREUNDLICH TROCKEN SONNE VIEL",
        "num_frames": 98
    },
    {
        "id": "phoenix_train_004",
        "signer": "Signer01",
        "text": "die temperaturen erreichen werte von zwölf grad an den alpen",
        "gloss": "TEMPERATUR GRAD ZWOELF ALPEN REGION",
        "num_frames": 105
    },
    {
        "id": "phoenix_train_005",
        "signer": "Signer04",
        "text": "der wind weht schwach bis mäßig aus nördlichen richtungen",
        "gloss": "WIND WEHEN SCHWACH MAESSIG NORD RICHTUNG",
        "num_frames": 88
    },
    {
        "id": "phoenix_train_006",
        "signer": "Signer02",
        "text": "in der nacht sinken die werte auf fünf bis minus zwei grad",
        "gloss": "NACHT TEMPERATUR SINKEN FUENF MINUS ZWEI GRAD",
        "num_frames": 120
    },
    {
        "id": "phoenix_dev_001",
        "signer": "Signer05",
        "text": "und nun die wettervorhersage für morgen donnerstag den fünfzehnten oktober",
        "gloss": "JETZT WETTER MORGEN DONNERSTAG FUNFZEHN OKTOBER",
        "num_frames": 110
    },
    {
        "id": "phoenix_dev_002",
        "signer": "Signer03",
        "text": "am donnerstag im norden regnerisch im süden oft heiter",
        "gloss": "DONNERSTAG NORD REGEN SUED OFT HEITER",
        "num_frames": 125
    },
    {
        "id": "phoenix_test_001",
        "signer": "Signer04",
        "text": "und nun die wettervorhersage für morgen freitag den fünfzehnten oktober",
        "gloss": "JETZT WETTER MORGEN FREITAG FUNFZEHN OKTOBER",
        "num_frames": 118
    },
    {
        "id": "phoenix_test_002",
        "signer": "Signer05",
        "text": "am freitag ist es in der nähe der küsten und ganz im süden teilweise stark bewölkt",
        "gloss": "FREITAG KUESTE REGION SUED TEILWEISE WOLKIG",
        "num_frames": 135
    },
    {
        "id": "phoenix_test_003",
        "signer": "Signer01",
        "text": "am wochenende meist sonnig und trocken bei steigenden temperaturen",
        "gloss": "WOCHENENDE MEIST SONNE TROCKEN TEMPERATUR STEIGEN",
        "num_frames": 102
    }
]

def generate_mock_phoenix_dataset(target_dir="./data/raw/phoenix2014t", num_dummy_frames=16):
    """
    Creates mock directory structure for PHOENIX-2014T dataset:
      - annotations/train.json
      - annotations/dev.json
      - annotations/test.json
      - features/ or frame folders with mock images
    """
    os.makedirs(os.path.join(target_dir, "annotations"), exist_ok=True)
    os.makedirs(os.path.join(target_dir, "frames"), exist_ok=True)

    train_samples = [s for s in SAMPLE_SENTENCES if "train" in s["id"]]
    dev_samples = [s for s in SAMPLE_SENTENCES if "dev" in s["id"]]
    test_samples = [s for s in SAMPLE_SENTENCES if "test" in s["id"]]

    for split, samples in [("train", train_samples), ("dev", dev_samples), ("test", test_samples)]:
        ann_path = os.path.join(target_dir, "annotations", f"{split}.json")
        with open(ann_path, "w", encoding="utf-8") as f:
            json.dump(samples, f, indent=2, ensure_ascii=False)

        # Generate sample frame images for each sample
        for sample in samples:
            sample_dir = os.path.join(target_dir, "frames", sample["id"])
            os.makedirs(sample_dir, exist_ok=True)
            # Create a few synthetic frames (e.g. 16 frames with moving circle to simulate gesture)
            for idx in range(num_dummy_frames):
                img_path = os.path.join(sample_dir, f"frame_{idx:05d}.jpg")
                if not os.path.exists(img_path):
                    # Create 224x224 RGB image with gradient & moving articulator circle
                    arr = np.zeros((224, 224, 3), dtype=np.uint8)
                    arr[:, :, 0] = 50 + int(idx * 5) % 150 # Background shift
                    arr[:, :, 1] = 70
                    arr[:, :, 2] = 100
                    # Simulate moving hand articulator (motion difference)
                    cx = int(80 + 60 * np.sin(idx * 0.4))
                    cy = int(100 + 40 * np.cos(idx * 0.4))
                    radius = 20
                    y, x = np.ogrid[:224, :224]
                    mask = (x - cx)**2 + (y - cy)**2 <= radius**2
                    arr[mask] = [230, 200, 160] # Skin tone articulator
                    img = Image.fromarray(arr)
                    img.save(img_path)

    print(f"[Dataset] Mock PHOENIX-2014T generated successfully in: {target_dir}")
    return target_dir

if __name__ == "__main__":
    generate_mock_phoenix_dataset()


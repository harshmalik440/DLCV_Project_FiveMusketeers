# DynST-VLM – Model Checkpoints

This folder contains the model checkpoints used in the DynST-VLM project.

Due to their large file sizes, the checkpoint files are hosted externally on Google Drive instead of being committed directly to this GitHub repository.

## 1. Checkpoint Downloads

| Model | Checkpoint File | Download |
|---|---|---|
| Baseline | `baseline_best.pt` | [Download Baseline](https://drive.google.com/file/d/1UShvNbWoFgZuWRqZDa6gr_60aoSNqu-N/view?usp=drive_link) |
| Proposed DynST-VLM | `proposed_best.pt` | [Download Proposed](https://drive.google.com/file/d/1mmk5zncVWFWcK-0q2YhwfOGP8dFSWTCM/view?usp=drive_link) |

## 2. Installation Instructions

1. Open the appropriate Google Drive link.
2. Download the required checkpoint file.
3. Place the downloaded `.pt` files directly inside this `checkpoints/` folder.

Expected directory structure:

```text
DynST-VLM/
│
├── checkpoints/
│   ├── README.md
│   ├── baseline_best.pt
│   └── proposed_best.pt
│
├── configs/
├── models/
├── scripts/
├── data/
├── results/
├── logs/
├── requirements.txt
└── README.md
```

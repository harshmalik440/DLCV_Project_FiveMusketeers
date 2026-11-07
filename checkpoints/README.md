# DynST-VLM – Model Checkpoints

This folder contains the model checkpoints used in the DynST-VLM project.

Due to their large file sizes, the checkpoint files are hosted externally on Google Drive instead of being committed directly to this GitHub repository.

## 1. Checkpoint Downloads

| Model | Checkpoint File | Download |
|---|---|---|
| Baseline | `baseline_best.pt` | [Download Baseline](PASTE_BASELINE_GOOGLE_DRIVE_LINK) |
| Proposed DynST-VLM | `proposed_best.pt` | [Download Proposed](PASTE_PROPOSED_GOOGLE_DRIVE_LINK) |

## 2. Installation Instructions

1. Open the appropriate Google Drive link.
2. Download the required checkpoint file.
3. Place the downloaded `.pt` files directly inside this `checkpoints/` folder.
4. Ensure that the filenames match those expected by the inference and evaluation scripts.

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

## 3. Important Notes

- Checkpoints are hosted externally because of their large file sizes.
- Google Drive access should be set to Anyone with the link – Viewer.
- Ensure that the downloaded files are complete before running inference.
- Do not rename the checkpoint files unless the corresponding paths in the code are updated.

## 4. Project Information

**Project:** DynST-VLM – Dynamic Spatio-Temporal Difference Connector with Contrastive Semantic Anchoring for Sign Language Translation.

**Team:** Team 26

Refer to the [main project README](../README.md) for installation, training, evaluation, and inference instructions.

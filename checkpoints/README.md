# Model Checkpoints Directory

This directory contains trained and initialized model weights for the Sign Language Translation evaluation pipelines.

## Directory Structure
```
checkpoints/
├── baseline/
│   └── baseline_best.pt        # SmolVLM2-2.2B Baseline with Static MLP Connector
└── proposed/
    └── proposed_best.pt        # DynST-VLM with Motion-Difference & Articulator Saliency
```

## Checkpoint Format
Each `.pt` checkpoint file is a serialized PyTorch dictionary containing:
* `epoch`: Training epoch at checkpoint creation.
* `model_state_dict`: Complete weights for connector, LoRA adapters, and language model projections.
* `vocab`: Tokenizer token-to-ID vocabulary mapping.
* `loss`: Training / validation loss at optimal checkpoint.
* `model_type`: `"baseline"` or `"proposed"`.
* `description`: Model architecture and training specification.

## Generation
If initializing reference checkpoints from scratch:
```bash
python scripts/generate_sample_checkpoints.py
```

## Evaluation Commands
To evaluate either checkpoint against the test benchmark:
```bash
# Evaluate baseline
python scripts/evaluate.py --checkpoint checkpoints/baseline/baseline_best.pt --split test

# Evaluate proposed DynST-VLM
python scripts/evaluate.py --checkpoint checkpoints/proposed/proposed_best.pt --split test
```


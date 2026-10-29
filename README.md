# Remote Sensing Image Captioning — NWPU-Caption Benchmark

A modular, production-ready Deep Learning framework for remote sensing image recognition and captioning using the **NWPU-Caption** benchmark dataset (NWPU-RESISC45 images with five human captions per image).

---

## Architecture Overview

```
                      ┌───────────────────────────────────────────────┐
                      │             Input Satellite Image             │
                      │               (256 x 256 x 3)                 │
                      └──────────────────────┬────────────────────────┘
                                             │
                                             ▼
                      ┌───────────────────────────────────────────────┐
                      │    ConvNeXt-Tiny (ImageNet-12k ft 1k)         │
                      │    Frozen Backbone + Trainable LoRA PEFT      │
                      │         (r=16, alpha=32 on fc1, fc2)          │
                      └──────────────────────┬────────────────────────┘
                                             │
                                             ▼
                      ┌───────────────────────────────────────────────┐
                      │          Spatial Grid Visual Tokens           │
                      │               (64 x 768 tokens)               │
                      └──────────────┬────────────────┬───────────────┘
                                     │                │
            ┌────────────────────────┘                └────────────────────────┐
            │ Mean-Pooled Features                                             │
            ▼                                                                  ▼
┌───────────────────────┐                                          ┌───────────────────────┐
│ State Initialization  │                                          │  Bahdanau Attention   │
│  h_0 = tanh(W_h * V)  │                                          │ e_ti = v_a tanh(...)  │
│  c_0 = tanh(W_c * V)  │                                          │ alpha_t = softmax(e)  │
└───────────┬───────────┘                                          │  z_t = sum(alpha * V) │
            │                                                      └───────────┬───────────┘
            │                                                                  │
            └─────────────────────────┬────────────────────────────────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────────────────────┐
                      │           Attention LSTM Decoder              │
                      │    Input: [Word Embedding; Beta-Gated z_t]    │
                      │           Hidden Dimension: 512               │
                      └──────────────────────┬────────────────────────┘
                                             │
                                             ▼
                      ┌───────────────────────────────────────────────┐
                      │       Doubly Stochastic Regularization        │
                      │   Loss = CrossEntropy + lambda * (1 - sum a)^2 │
                      └───────────────────────────────────────────────┘
```

- **Image Encoder**: `convnext_tiny.in12k_ft_in1k` via `timm` (ImageNet-12k pretraining fine-tuned on 1k, Top-1 84.19%).
- **PEFT Adaptation**: LoRA low-rank adaptation on pointwise MLP layers (`mlp.fc1`, `mlp.fc2`) with rank $r=16$, $\alpha=32$. Only ~0.6M parameters train while the 28M backbone parameters remain frozen.
- **Attention Mechanism**: Bahdanau additive spatial attention mapping the $8 \times 8 = 64$ spatial grid tokens and decoder hidden state to attention weights $\alpha_t$.
- **Language Decoder**: LSTM Cell with Show, Attend and Tell context gating ($\beta$-gate) and doubly stochastic attention penalty ($\lambda=1.0$).
- **Benchmark Evaluation**: Pure Python implementation of BLEU-1, BLEU-2, BLEU-3, BLEU-4, and ROUGE-L, alongside multi-panel spatial attention heatmap visualizations.

---

## Project Structure

```
Image_Recognition/
├── dataset/                    # Local dataset storage (cleaned & deduplicated)
│   ├── dataset_nwpu.json       # 31,491 images, 157,455 captions (exact & leakage-free)
│   ├── vocab.json              # 2,661 word-level vocabulary tokens (built from train)
│   ├── images/                 # 45 class folders, 31,491 images (256x256 RGB)
│   └── README.md               # Dataset statistics, schema, and deduplication audit
├── notebooks/                  # Interactive experimentation
│   └── kaggle_training.ipynb   # Ready-to-run GPU training notebook for Kaggle
├── reports/                    # Quality analysis, audits, and evaluation metrics
│   ├── validation_report.md    # Deduplication and quality audit report
│   └── visualizations/         # Word-by-word spatial attention heatmap plots
├── scripts/                    # Automation and pipeline scripts
│   ├── train.py                # PyTorch training pipeline with AMP & checkpointing
│   ├── evaluate.py             # Evaluation runner for BLEU-1..4 and ROUGE-L
│   ├── kaggle_standalone.py    # Zero-dependency single-paste script for Kaggle cells
│   ├── build_kaggle_notebook.py# Generator for kaggle_training.ipynb
│   ├── build_vocab.py          # Builds and verifies train-split vocabulary
│   ├── deduplicate_dataset.py  # Dataset deduplication & leakage resolution
│   └── validate_dataset.py     # Standalone validation and audit runner
├── src/
│   └── image_recognition/      # Core package library
│       ├── __init__.py         # Public package exports
│       ├── dataset.py          # NWPUCaptionItem, NWPUTorchDataset, collate_fn
│       ├── model.py            # ConvNeXtEncoder, BahdanauAttention, LSTMDecoder, NWPUCaptioner
│       ├── metrics.py          # BLEU-1..4, ROUGE-L sentence & corpus evaluators
│       ├── visualizer.py       # Spatial attention heatmap overlay visualizer
│       ├── validator.py        # Perceptual hash and cross-split leakage detector
│       └── vocab.py            # Word-level vocabulary tokenizer & indexer
├── tests/                      # Automated test suite
│   ├── test_dataset.py         # Dataset counts, splits, and collate_fn tests
│   ├── test_metrics.py         # BLEU-1..4 and ROUGE-L mathematical unit tests
│   ├── test_model.py           # Forward, loss, LoRA gradient backprop, & generation tests
│   ├── test_validator.py       # Perceptual hash and distance verification
│   └── test_vocab.py           # Vocabulary special tokens, encode, and decode tests
├── pyproject.toml              # Project metadata and dependencies (uv)
└── uv.lock                     # Deterministic dependency lockfile
```

---

## Installation & Setup

Install dependencies and synchronize the environment using `uv`:
```bash
uv sync
```

---

## Training

### 1. Training on Kaggle (Recommended)
1. **Upload Dataset**: Upload the `dataset/` directory to Kaggle as a private dataset (e.g. named `nwpu-caption`).
2. **Execution Options**:
   - **Option A (Notebook)**: Upload `notebooks/kaggle_training.ipynb` to Kaggle, attach the dataset, select GPU (T4 x2 or P100), and click "Run All".
   - **Option B (Copy-Paste)**: Open a blank Kaggle notebook, copy-paste the entire contents of `scripts/kaggle_standalone.py` into a single code cell, attach your dataset, and run.
3. Path auto-detection will automatically discover `/kaggle/input/.../dataset_nwpu.json`, train with Automatic Mixed Precision (AMP), save the best checkpoint to `/kaggle/working/checkpoints/best_model.pt`, evaluate BLEU-1..4 and ROUGE-L on the test split, and output attention heatmaps.

### 2. Local Training
```bash
uv run python scripts/train.py --batch_size 32 --epochs 15
```

Key arguments:
- `--epochs`: Number of epochs (default: `15`).
- `--batch_size`: Batch size per step (default: `32`).
- `--lr_encoder`: Learning rate for LoRA adapter (default: `1e-4`).
- `--lr_decoder`: Learning rate for Attention LSTM (default: `4e-4`).
- `--no_amp`: Disable Automatic Mixed Precision.

---

## Evaluation & Qualitative Inspection

Evaluate a saved checkpoint on the test split (3,148 images with 5 reference captions each):
```bash
uv run python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --split test --num_visualizations 5
```

Output:
- Calculates corpus-level **BLEU-1, BLEU-2, BLEU-3, BLEU-4**, and **ROUGE-L**.
- Saves evaluation results in `reports/evaluation_test_results.json` and a Markdown summary in `reports/evaluation_test_report.md`.
- Generates word-by-word spatial attention heatmap images in `reports/visualizations/`.

---

## Running the Automated Test Suite

Run the full verification test suite:
```bash
uv run python tests/test_vocab.py
uv run python tests/test_dataset.py
uv run python tests/test_model.py
uv run python tests/test_metrics.py
uv run python tests/test_validator.py
```

---

## Dataset Quality & Hygiene Summary

- **Total Images**: 31,491 clean images (157,455 captions).
- **Splits**: 25,193 train (80.0%), 3,150 val (10.0%), 3,148 test (10.0%).
- **Leakage Resolution**:
  - Deduplicated 7 within-split duplicates in `train` (6 byte-identical, 1 perceptual-identical).
  - Resolved 2 cross-split duplicate pairs by removing `airport_635.jpg` and `palace_669.jpg` from `test`.
  - Zero exact image duplicates, zero perceptual duplicates, and zero cross-split leakage remain.

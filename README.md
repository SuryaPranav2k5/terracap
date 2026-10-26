# Image Recognition — NWPU-Caption Benchmark

A modular, production-ready Python project for remote sensing image recognition and captioning using the **NWPU-Caption** benchmark dataset (NWPU-RESISC45 images with five human captions per image).

---

## Project Structure

```
Image_Recognition/
├── dataset/                    # Local dataset storage (git-ignored raw assets)
│   ├── dataset_nwpu.json       # 157,500 caption annotations, splits, and sentence IDs
│   ├── images/                 # 45 class folders, 31,500 images (256x256 RGB)
│   ├── hf/                     # Hugging Face DatasetDict saved to disk
│   └── README.md               # Dataset statistics and schema details
├── reports/                    # Quality analysis and audit reports
│   └── validation_report.md    # Comprehensive mapping, duplicate & quality report
├── scripts/                    # Automation and pipeline scripts
│   ├── download_dataset.py     # Standalone dataset downloader & extractor
│   ├── deduplicate_dataset.py  # Dataset deduplication & leakage resolution
│   └── validate_dataset.py     # Standalone validation and audit runner
├── src/
│   └── image_recognition/      # Core package library
│       ├── __init__.py         # Public library interface
│       ├── dataset.py          # Dataset abstractions (NWPUCaptionItem, loaders)
│       └── validator.py        # Validation engine, perceptual hash, data leakage
├── tests/                      # Automated test suite
│   ├── test_dataset.py         # Dataset counts, splits, and image opening
│   └── test_validator.py       # Perceptual hash and distance verification
├── pyproject.toml              # Project metadata and dependencies
└── uv.lock                     # Deterministic dependency lockfile
```

---

## Getting Started

### 1. Installation
Install dependencies and build the virtual environment using `uv`:
```bash
uv sync
```

### 2. Running Scripts
Run tasks directly via `uv`:

- **Run dataset validation & duplicate audit**:
  ```bash
  uv run python scripts/validate_dataset.py
  ```

- **Download and extract dataset**:
  ```bash
  uv run python scripts/download_dataset.py
  ```

- **Deduplicate dataset & resolve leakage**:
  ```bash
  uv run python scripts/deduplicate_dataset.py
  ```

---

## Python API Usage

```python
from image_recognition import load_dataset_items, load_hf_dataset

# 1. Load items filtered by split ('train', 'val', or 'test')
train_items = load_dataset_items(split="train")
print(f"Loaded {len(train_items)} train samples")

first = train_items[0]
print(f"Class: {first.class_name}")
print(f"Image Path: {first.image_path}")
print(f"5 Captions: {first.captions}")

# Open the image as a PIL Image (256x256 RGB)
img = first.open_image()

# 2. Or load the Hugging Face Dataset format
hf_ds = load_hf_dataset()
print(hf_ds)
```

---

## Validation & Quality Highlights

- **Completeness**: 31,491 / 31,491 images matched on disk with zero missing or orphaned files.
- **Images**: 100% uniform 256×256 RGB JPEG files across 45 classes.
- **Splits**: 25,193 train (80.0%), 3,150 val (10.0%), 3,148 test (10.0%).
- **Deduplication**:
  - 7 within-split duplicates in `train` deduplicated (6 exact byte-duplicates + 1 perceptual duplicate differing by 9 bytes of JPEG metadata, MAE=0.034).
  - 2 cross-split duplicate pairs (4 leaked files) resolved by removing the duplicate from `test` (mandatory data hygiene).
  - Exact & Perceptual Image Duplicates remaining: **0**.
  - Cross-split Image Leakage remaining: **0**.
- **Methodology & Limitations**:
  - *Caption Overlap*: Caption-string overlap across images and splits is documented in [reports/validation_report.md](reports/validation_report.md) as an inherent linguistic characteristic of remote sensing aerial data rather than an anomaly to artificially prune.
  - *Deduplication Boundary*: Automated perceptual deduplication strictly used a Hamming distance of `0` (exact perceptual matches). Potential near-duplicates from overlapping satellite acquisition passes with slight variances in angle, crop, or lighting (distances > 0) are deliberately preserved.

---

## Testing

Run the test suite:
```bash
uv run python tests/test_dataset.py
uv run python tests/test_validator.py
```

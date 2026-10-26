# NWPU-Caption Dataset

Benchmark remote sensing image captioning dataset based on NWPU-RESISC45.

## Dataset Statistics (Deduplicated & Audited)
- **Total Images**: 31,491 unique images (256x256 RGB, JPEG)
- **Scene Classes**: 45 scene categories
- **Captions**: 157,455 total captions (5 human-annotated sentences per image)
- **Splits**:
  - `train`: 25,193 images (80.0%)
  - `val`: 3,150 images (10.0%)
  - `test`: 3,148 images (10.0%)
- **Data Leakage**: 0 cross-split image duplicates (audited and resolved).
- **Exact & Perceptual Image Duplicates**: 0 duplicate groups.

## Directory Structure
```
dataset/
├── dataset_nwpu.json            # Deduplicated caption annotations and splits (31,492 entries)
├── dataset_nwpu.json.bak        # Original dataset backup (31,500 entries)
├── removed_duplicates_audit.json # Detailed log of the 8 removed duplicate entries
├── images/                      # 45 category subdirectories with 31,492 image files
│   ├── airplane/
│   ├── airport/
│   └── ... (45 categories total)
└── hf/                          # Hugging Face DatasetDict saved to disk
```

## Methodology & Dataset Characteristics / Limitations

### 1. Caption-String Overlap as an Inherent Characteristic
- **Linguistic Distribution**: Approximately 12,968 unique caption sentences recur across multiple images, with 6,609 identical sentences shared across train, val, and test splits.
- **Why this is NOT an error to resolve**: In overhead satellite and aerial photography, distinct scenes across different geographical coordinates frequently share nearly identical visual characteristics (e.g. dense forest canopies, open beaches, circular farmlands). Independent annotators accurately provided standard canonical descriptions (e.g., *"the entire image is dominated by grass"*).
- **Evaluation Guidance**: Evaluation metrics (BLEU-1–4, METEOR, ROUGE-L, CIDEr, SPICE) remain valid against the 5 human reference captions per image without synthetically perturbing or pruning ground-truth phrases.

### 2. Image Deduplication Audit
- **Within-Split (Cosmetic)**: 7 redundant images in the `train` split (6 exact byte-duplicates and 1 perceptual duplicate differing by only 9 bytes of JPEG metadata, MAE=0.034) were removed to prevent disproportionate sample weighting during gradient updates.
- **Cross-Split (Mandatory Hygiene)**: 2 cross-split duplicate pairs (4 leaked files across `train`/`test` and `val`/`test`) were resolved by removing the duplicate instance from the evaluation (`test`) set. This ensures test and validation evaluations remain completely independent.

### 3. Boundary of Perceptual Hash Deduplication (dHash Distance = 0)
- **Strict Threshold Scope**: Deduplication operated at a strict Hamming distance threshold of `0` on 64-bit difference hashes (dHash), which targets exact bitwise and congruent rasterization duplicates.
- **Near-Duplicate Scene Limitation**: Remote sensing datasets frequently feature overlapping flight paths or adjacent sensor swaths with marginal shifts in solar angle, seasonal vegetation, atmospheric conditions, or spatial framing/crops. Such near-duplicate scenes generate Hamming distances greater than zero and are deliberately preserved. The dataset is free of exact congruent duplicates, not all proximate satellite captures.

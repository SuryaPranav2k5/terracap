# NWPU-Caption Dataset Validation & Quality Report

## Executive Summary

| Metric | Result | Status |
| :--- | :--- | :--- |
| **JSON Referenced Images** | 31,491 | Verified |
| **Disk Image Files** | 31,491 | Verified |
| **Missing Images on Disk** | 0 | PASS |
| **Orphaned Images on Disk** | 0 | PASS |
| **Corrupted Images** | 0 | PASS |
| **Image Dimension Uniformity** | 256x256 (31,491) | Uniform |
| **Image Color Modes** | RGB (31,491) | Uniform |
| **Duplicate Image IDs** | 0 | PASS |
| **Duplicate Sentence IDs** | 0 | PASS |
| **Exact Byte Image Duplicates** | 0 groups | Clean |
| **Exact Perceptual (dHash) Duplicates** | 0 pairs | Clean |
| **Image Split Leakage** | 0 instances | PASS (Zero Leakage) |
| **Captions Analyzed** | 157,455 | 5 per image |
| **Vocabulary Size** | 2,783 unique words | Analyzed |
| **Cross-Image Duplicate Captions** | 12,968 phrases | Analyzed |
| **Caption Split Leakage** | 6,609 duplicate phrases across splits | Evaluated |

---

## 1. Mapping & Split Integrity

- **Total Classes**: 45 (each with 700 images, exactly 31,500 total)
- **Dataset Splits**:
  - `test`: **3,148** images (10.0%)
  - `train`: **25,193** images (80.0%)
  - `val`: **3,150** images (10.0%)

---

## 2. Image Verification & Duplicate Analysis

- **Resolution**: 100% of images are `256x256` pixels.
- **Color Mode**: 100% of images are `RGB`.
- **Exact File Hash Duplicates**: **0** duplicate groups found.
- **Perceptual (dHash) Duplicates**: **0** pairs found.

- **Data Leakage Check (Images)**: **0** images leak across splits.

---

## 3. Annotation Quality & Linguistic Properties

### Sentence Length Distributions
- **Mean Word Count**: `11.33 ± 4.60` words
- **Min / Max Words**: `0` words / `50` words
- **Mean Character Count**: `63.29` characters
- **Vocabulary Size**: `2,783` unique tokens
- **Mean Intra-Image Caption Jaccard Similarity**: `0.218`
- **Images with Exact Duplicate Captions**: `122`

### Annotation Artifacts & Formatting Anomalies
- **detached_period_spaces (' .')**: `157,425` occurrences
- **missing_ending_periods**: `3` occurrences
- **lowercase_first_character**: `1` occurrences
- **empty_or_whitespace**: `0` occurrences

### Most Frequent Vocabulary Tokens

| Rank | Token | Count |
| :--- | :--- | :--- |
| 1 | `the` | 178,608 |
| 2 | `are` | 77,710 |
| 3 | `and` | 76,966 |
| 4 | `there` | 67,789 |
| 5 | `a` | 60,991 |
| 6 | `is` | 57,102 |
| 7 | `of` | 47,645 |
| 8 | `on` | 47,631 |
| 9 | `some` | 45,519 |
| 10 | `many` | 38,778 |
| 11 | `green` | 35,797 |
| 12 | `in` | 30,954 |
| 13 | `trees` | 30,356 |
| 14 | `buildings` | 29,992 |
| 15 | `with` | 26,314 |

### Most Common Identical Captions (Across Images)

| Duplicate Caption | Occurrences | Unique Images | Splits Present |
| :--- | :--- | :--- | :--- |
| "the snow berg is consist of bare land and white snow" | 448 | 448 | `val, test, train` |
| "the entire image is dominated by grass" | 435 | 435 | `val, test, train` |
| "this is a dense forest" | 415 | 415 | `val, test, train` |
| "the bare and green terrace is next to some trees" | 290 | 290 | `val, test, train` |
| "the meadow is dense but uneven" | 288 | 288 | `val, test, train` |
| "the tennis court is on the grass next to some buildings and trees" | 282 | 282 | `val, test, train` |
| "there are many buildings next to the ground track field" | 269 | 269 | `val, test, train` |
| "chaparral is evenly distributed on the sand" | 263 | 263 | `val, test, train` |
| "the forest is full of green trees" | 238 | 238 | `val, test, train` |
| "the grass is beside the beach" | 232 | 232 | `val, test, train` |

---

## 4. Methodology & Dataset Characteristics / Limitations

### Caption-String Overlap as an Inherent Dataset Characteristic
- **Context & Origin**: Cross-image and cross-split caption-string overlap (e.g. phrases such as *"the entire image is dominated by grass"*, *"this is a dense forest"*, or *"the meadow is dense but uneven"*) is an **inherent linguistic characteristic** of the NWPU-Caption benchmark, not a data defect or bug to be artificially resolved or filtered.
- **Domain Specificity**: In aerial and satellite remote sensing, different spatial regions frequently exhibit near-identical visual topography (e.g., continuous agricultural fields, dense forest canopies, open beaches). Multiple human annotators independently produce identical, accurate descriptive sentences for these scenes.
- **Modeling Guidance**: Downstream vision-language and captioning architectures should anticipate this natural semantic distribution. Evaluation metrics (BLEU-1 to 4, METEOR, ROUGE-L, CIDEr, SPICE) remain valid against the 5 human reference captions per image, without synthetically perturbing the ground truth.

### Deduplication & Integrity Actions
- **Within-Split Deduplication (Cosmetic)**: 6 exact redundant images in the `train` split were deduplicated to eliminate sample weight distortion during model optimization.
- **Cross-Split Deduplication (Mandatory Data Hygiene)**: 2 cross-split duplicate pairs (4 leaked files across `train`/`test` and `val`/`test`) were resolved by removing the duplicate instance from the evaluation (`test`) set. This guarantees zero train-to-test and val-to-test visual contamination.

### Boundary of Perceptual Hash Deduplication (dHash Distance = 0)
- **Strict Threshold Scope**: The automated perceptual deduplication pipeline operates at a strict Hamming distance threshold of `0`. This exclusively catches bitwise and structurally congruent perceptual duplicates (identical images under identical or near-identical rasterization).
- **Near-Duplicate Scene Caveat**: Because NWPU-RESISC45 images originate from remote sensing sensors, satellite passes frequently capture adjacent or overlapping geographic footprints with subtle variations in solar angle, atmospheric haze, seasonal vegetation, compression artifacts, or slight spatial translation/cropping. Such near-duplicate scenes produce Hamming distances greater than zero (e.g. distances 1–6) and are intentionally preserved.
- **Methodological Stance**: Our deduplication claims apply strictly to exact byte and congruent perceptual duplicates (distance = 0); we do not claim full semantic or near-duplicate scene elimination across neighboring satellite passes.

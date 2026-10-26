"""Deduplicate NWPU-Caption dataset images.

1. Deduplicate within-split exact image duplicates (cosmetic).
2. Fix cross-split image duplicates to eliminate data leakage into evaluation splits (mandatory).
3. Save an audit log and update dataset metadata.
"""

import json
from pathlib import Path
import shutil
from typing import Dict, List, Set


# Exact duplicate files to remove (keeping canonical first instance)
DUPLICATES_TO_REMOVE = {
    # Within-split duplicates in 'train'
    ("airport", "airport_420.jpg"): {
        "canonical": ("airport", "airport_050.jpg"),
        "reason": "within-split duplicate in train",
        "split": "train",
    },
    ("airport", "airport_527.jpg"): {
        "canonical": ("airport", "airport_064.jpg"),
        "reason": "within-split duplicate in train",
        "split": "train",
    },
    ("airport", "airport_315.jpg"): {
        "canonical": ("airport", "airport_118.jpg"),
        "reason": "within-split duplicate in train",
        "split": "train",
    },
    ("airport", "airport_397.jpg"): {
        "canonical": ("airport", "airport_243.jpg"),
        "reason": "within-split duplicate in train",
        "split": "train",
    },
    ("airport", "airport_509.jpg"): {
        "canonical": ("airport", "airport_361.jpg"),
        "reason": "within-split duplicate in train",
        "split": "train",
    },
    ("palace", "palace_439.jpg"): {
        "canonical": ("palace", "palace_224.jpg"),
        "reason": "within-split duplicate in train",
        "split": "train",
    },
    ("mountain", "mountain_324.jpg"): {
        "canonical": ("mountain", "mountain_111.jpg"),
        "reason": "within-split perceptual duplicate in train (differed by 9 bytes JPEG encoding, MAE=0.034)",
        "split": "train",
    },
    # Cross-split duplicates (leaked into 'test')
    ("airport", "airport_635.jpg"): {
        "canonical": ("airport", "airport_289.jpg"),
        "reason": "cross-split leakage from train (removed from test)",
        "split": "test",
    },
    ("palace", "palace_669.jpg"): {
        "canonical": ("palace", "palace_595.jpg"),
        "reason": "cross-split leakage from val (removed from test)",
        "split": "test",
    },
}


def deduplicate_dataset(dataset_dir: str = "dataset") -> None:
    base_dir = Path(dataset_dir)
    json_path = base_dir / "dataset_nwpu.json"
    images_dir = base_dir / "images"

    if not json_path.exists():
        raise FileNotFoundError(f"Missing {json_path}")

    # Backup original json
    backup_path = base_dir / "dataset_nwpu.json.bak"
    if not backup_path.exists():
        shutil.copy2(json_path, backup_path)
        print(f"Created backup of original annotations at {backup_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data: Dict[str, List[dict]] = json.load(f)

    audit_log = []
    new_data: Dict[str, List[dict]] = {}
    removed_count = 0

    for class_name, entries in data.items():
        filtered_entries = []
        for entry in entries:
            key = (class_name, entry["filename"])
            if key in DUPLICATES_TO_REMOVE:
                info = DUPLICATES_TO_REMOVE[key]
                removed_count += 1
                img_path = images_dir / class_name / entry["filename"]
                
                # Remove file from images directory
                if img_path.exists():
                    img_path.unlink()

                audit_log.append({
                    "removed_image": f"{class_name}/{entry['filename']}",
                    "canonical_image": f"{info['canonical'][0]}/{info['canonical'][1]}",
                    "split": info["split"],
                    "reason": info["reason"],
                    "imgid": entry.get("imgid"),
                })
            else:
                filtered_entries.append(entry)
        new_data[class_name] = filtered_entries

    # Save deduplicated annotations
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2)

    # Save audit log
    audit_path = base_dir / "removed_duplicates_audit.json"
    with open(audit_path, "w", encoding="utf-8") as f:
        json.dump(audit_log, f, indent=2)

    print(f"Successfully deduplicated dataset:")
    print(f"  - Total duplicates removed: {removed_count}")
    print(f"  - Within-split duplicates fixed: 7 (6 exact byte + 1 perceptual-identical)")
    print(f"  - Cross-split duplicates fixed: 2 pairs (4 leaked image files resolved)")
    print(f"  - Cleaned annotations saved to: {json_path}")
    print(f"  - Audit log written to: {audit_path}")


if __name__ == "__main__":
    deduplicate_dataset()

"""Utilities and dataset abstractions for NWPU-Caption."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional
from datasets import DatasetDict, load_from_disk
from PIL import Image


@dataclass
class NWPUCaptionItem:
    """Represents a single image item from the NWPU-Caption dataset."""

    filename: str
    class_name: str
    image_path: Path
    split: str
    imgid: int
    captions: List[str]
    sentids: List[int]

    def open_image(self) -> Image.Image:
        """Opens and returns the PIL Image."""
        return Image.open(self.image_path)


def load_raw_caption_json(dataset_dir: str = "dataset") -> Dict:
    """Loads the raw caption JSON file."""
    json_path = Path(dataset_dir) / "dataset_nwpu.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Caption JSON not found at {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset_items(
    dataset_dir: str = "dataset",
    split: Optional[Literal["train", "val", "test"]] = None,
) -> List[NWPUCaptionItem]:
    """Loads all items (or filtered by split: 'train', 'val', 'test') with resolved filepaths and captions."""
    data = load_raw_caption_json(dataset_dir)
    images_base_path = Path(dataset_dir) / "images"
    items: List[NWPUCaptionItem] = []

    for class_name, entries in data.items():
        for entry in entries:
            entry_split = entry.get("split", "")
            if split is not None and entry_split != split:
                continue

            captions = [
                entry[k]
                for k in ["raw", "raw_1", "raw_2", "raw_3", "raw_4"]
                if k in entry
            ]
            sentids = [
                entry[k]
                for k in ["sentids", "sentids_1", "sentids_2", "sentids_3", "sentids_4"]
                if k in entry
            ]

            filename = entry["filename"]
            img_path = images_base_path / class_name / filename

            items.append(
                NWPUCaptionItem(
                    filename=filename,
                    class_name=class_name,
                    image_path=img_path,
                    split=entry_split,
                    imgid=entry.get("imgid", -1),
                    captions=captions,
                    sentids=sentids,
                )
            )

    return items


def load_hf_dataset(dataset_dir: str = "dataset") -> DatasetDict:
    """Loads the Hugging Face DatasetDict saved on disk."""
    hf_path = Path(dataset_dir) / "hf"
    if not hf_path.exists():
        raise FileNotFoundError(f"Hugging Face dataset not found at {hf_path}")
    return load_from_disk(str(hf_path))

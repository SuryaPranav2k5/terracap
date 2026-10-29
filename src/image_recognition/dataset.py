"""Utilities and dataset abstractions for NWPU-Caption."""

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple, Union
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

from image_recognition.vocab import Vocabulary


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
        return Image.open(self.image_path).convert("RGB")


def resolve_dataset_dir(dataset_dir: Optional[str] = None) -> Path:
    """Auto-detects dataset directory (checks Kaggle input subdirectories vs local directory)."""
    if dataset_dir and Path(dataset_dir).exists():
        return Path(dataset_dir)

    # If running on Kaggle, search /kaggle/input for dataset_nwpu.json
    kaggle_root = Path("/kaggle/input")
    if kaggle_root.exists():
        # Check direct or 1-2 levels deep
        for candidate in [kaggle_root] + list(kaggle_root.iterdir()):
            if (candidate / "dataset_nwpu.json").exists():
                return candidate
            if (candidate / "dataset" / "dataset_nwpu.json").exists():
                return candidate / "dataset"

    # Fallback to local default
    local_path = Path("dataset")
    if local_path.exists() and (local_path / "dataset_nwpu.json").exists():
        return local_path
    return Path(dataset_dir or "dataset")



def load_raw_caption_json(dataset_dir: Optional[str] = None) -> Dict:
    """Loads the raw caption JSON file."""
    base_path = resolve_dataset_dir(dataset_dir)
    json_path = base_path / "dataset_nwpu.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Caption JSON not found at {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_dataset_items(
    dataset_dir: Optional[str] = None,
    split: Optional[Literal["train", "val", "test"]] = None,
) -> List[NWPUCaptionItem]:
    """Loads all items (or filtered by split) with resolved filepaths and captions."""
    base_path = resolve_dataset_dir(dataset_dir)
    data = load_raw_caption_json(str(base_path))
    images_base_path = base_path / "images"
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


def get_default_transforms(split: str = "train") -> Callable:
    """Returns standard ImageNet image transforms."""
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if split == "train":
        return T.Compose([
            T.Resize((256, 256)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
            normalize,
        ])
    return T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        normalize,
    ])


class NWPUTorchDataset(Dataset):
    """PyTorch Dataset for NWPU-Caption.

    When unroll_captions=False (recommended for fast training):
        Yields (image, list_of_5_encoded_captions). The ConvNeXt encoder runs
        only ONCE per image, expanding features across the 5 captions in collate_fn.
    When unroll_captions=True:
        Yields (image, single_encoded_caption) (legacy unrolled samples).
    In 'eval' mode:
        Yields (image, list_of_5_reference_strings, image_id).
    """

    def __init__(
        self,
        dataset_dir: Optional[str] = None,
        split: str = "train",
        vocab: Optional[Vocabulary] = None,
        transform: Optional[Callable] = None,
        mode: Optional[str] = None,
        unroll_captions: bool = False,
    ):
        self.base_dir = resolve_dataset_dir(dataset_dir)
        self.split = split
        self.mode = mode or ("train" if split == "train" else "eval")
        self.unroll_captions = unroll_captions
        self.transform = transform or get_default_transforms(split)

        # Load vocabulary
        if vocab is not None:
            self.vocab = vocab
        else:
            vocab_path = self.base_dir / "vocab.json"
            self.vocab = Vocabulary.load(vocab_path) if vocab_path.exists() else Vocabulary()

        items = load_dataset_items(str(self.base_dir), split=split)  # type: ignore

        if self.mode == "train":
            if self.unroll_captions:
                # Unroll 1 image -> 5 samples (legacy mode)
                self.samples: List[Tuple[Path, str, int]] = []
                for item in items:
                    for cap in item.captions:
                        self.samples.append((item.image_path, cap, item.imgid))
            else:
                # 1 image per item with all 5 captions (eliminates 5x encoder forward redundancy)
                self.items = items
        else:
            self.eval_items = items

    def __len__(self) -> int:
        if self.mode == "train":
            return len(self.samples) if self.unroll_captions else len(self.items)
        return len(self.eval_items)

    def __getitem__(self, idx: int):
        if self.mode == "train":
            if self.unroll_captions:
                img_path, caption, imgid = self.samples[idx]
                image = Image.open(img_path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                token_ids = self.vocab.encode(caption, add_special_tokens=True)
                return image, torch.tensor(token_ids, dtype=torch.long)
            else:
                item = self.items[idx]
                image = item.open_image()
                if self.transform:
                    image = self.transform(image)
                cap_tensors = [
                    torch.tensor(self.vocab.encode(c, add_special_tokens=True), dtype=torch.long)
                    for c in item.captions
                ]
                return image, cap_tensors
        else:
            item = self.eval_items[idx]
            image = item.open_image()
            if self.transform:
                image = self.transform(image)
            return image, item.captions, item.imgid


def caption_collate_fn(batch):
    """Pads captions and sorts descending by length.

    Supports both:
    1. Single-caption items: (image, tensor_caption) -> returns (images, targets, lengths, None)
    2. Multi-caption items: (image, list_of_5_caption_tensors) -> returns (images, targets, lengths, img_indices)
       allowing ConvNeXt to execute only ONCE per image while decoding all 5 captions.
    """
    first_item = batch[0]
    if isinstance(first_item[1], list):
        # Multi-caption 1-pass encoder mode
        images, multi_captions = zip(*batch)
        images = torch.stack(images, dim=0)

        flat_captions = []
        for img_idx, caps in enumerate(multi_captions):
            for cap in caps:
                flat_captions.append((cap, img_idx))

        # Sort all captions across the batch descending by length
        flat_captions.sort(key=lambda x: len(x[0]), reverse=True)

        sorted_caps = [c[0] for c in flat_captions]
        img_indices = torch.tensor([c[1] for c in flat_captions], dtype=torch.long)
        lengths = torch.tensor([len(c) for c in sorted_caps], dtype=torch.long)
        max_len = lengths.max().item()

        targets = torch.full((len(sorted_caps), max_len), fill_value=Vocabulary.PAD_IDX, dtype=torch.long)
        for i, cap in enumerate(sorted_caps):
            targets[i, : len(cap)] = cap

        return images, targets, lengths, img_indices

    # Legacy single-caption mode
    batch.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    lengths = torch.tensor([len(c) for c in captions], dtype=torch.long)
    max_len = lengths.max().item()

    targets = torch.full((len(captions), max_len), fill_value=Vocabulary.PAD_IDX, dtype=torch.long)
    for i, cap in enumerate(captions):
        targets[i, : len(cap)] = cap

    return images, targets, lengths, None


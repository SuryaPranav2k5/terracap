"""Image Recognition Package for NWPU-Caption."""

from image_recognition.dataset import (
    NWPUCaptionItem,
    load_dataset_items,
    load_hf_dataset,
    load_raw_caption_json,
)
from image_recognition.validator import (
    DatasetValidator,
    ValidationResults,
    compute_dhash,
    hamming_distance,
)

__all__ = [
    "NWPUCaptionItem",
    "load_dataset_items",
    "load_hf_dataset",
    "load_raw_caption_json",
    "DatasetValidator",
    "ValidationResults",
    "compute_dhash",
    "hamming_distance",
]

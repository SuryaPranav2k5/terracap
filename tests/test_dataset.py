"""Tests for NWPU-Caption dataset and PyTorch DataLoader."""

import torch
from torch.utils.data import DataLoader

from image_recognition.dataset import (
    NWPUTorchDataset,
    caption_collate_fn,
    load_dataset_items,
)
from image_recognition.vocab import Vocabulary


def test_dataset_counts():
    train_items = load_dataset_items(split="train")
    val_items = load_dataset_items(split="val")
    test_items = load_dataset_items(split="test")
    all_items = load_dataset_items()

    assert len(train_items) == 25193, f"Expected 25193 train items, got {len(train_items)}"
    assert len(val_items) == 3150, f"Expected 3150 val items, got {len(val_items)}"
    assert len(test_items) == 3148, f"Expected 3148 test items, got {len(test_items)}"
    assert len(all_items) == 31491, f"Expected 31491 total items, got {len(all_items)}"
    assert len(train_items) + len(val_items) + len(test_items) == 31491

    # Verify first item
    first = train_items[0]
    assert first.image_path.exists(), f"Image path does not exist: {first.image_path}"
    assert len(first.captions) == 5, f"Expected 5 captions, got {len(first.captions)}"

    img = first.open_image()
    assert img.size == (256, 256), f"Expected image size (256, 256), got {img.size}"


def test_torch_dataset_and_dataloader():
    vocab = Vocabulary.load("dataset/vocab.json")

    # 1. Test 1-pass multi-caption mode (unroll_captions=False, default)
    dataset_1pass = NWPUTorchDataset(split="train", vocab=vocab, mode="train", unroll_captions=False)
    assert len(dataset_1pass) == 25193, f"Expected 25193 unique images, got {len(dataset_1pass)}"

    loader_1pass = DataLoader(dataset_1pass, batch_size=4, shuffle=False, collate_fn=caption_collate_fn)
    images, targets, lengths, img_indices = next(iter(loader_1pass))

    # 4 images batch should produce 4 images and 4 * 5 = 20 captions
    assert images.shape == (4, 3, 256, 256), f"Unexpected image shape: {images.shape}"
    assert targets.shape[0] == 20, f"Expected 20 captions for 4 images, got {targets.shape[0]}"
    assert lengths.shape == (20,)
    assert img_indices is not None and img_indices.shape == (20,)
    assert (img_indices < 4).all() and (img_indices >= 0).all()
    # Verify lengths are sorted descending
    for i in range(len(lengths) - 1):
        assert lengths[i] >= lengths[i + 1]
    # Verify special tokens
    for i in range(20):
        assert targets[i, 0].item() == Vocabulary.START_IDX
        assert targets[i, lengths[i] - 1].item() == Vocabulary.END_IDX

    # 2. Test legacy unrolled mode (unroll_captions=True)
    dataset_unrolled = NWPUTorchDataset(split="train", vocab=vocab, mode="train", unroll_captions=True)
    assert len(dataset_unrolled) == 25193 * 5, f"Expected 125965 samples, got {len(dataset_unrolled)}"

    loader_unrolled = DataLoader(dataset_unrolled, batch_size=4, shuffle=False, collate_fn=caption_collate_fn)
    images_u, targets_u, lengths_u, img_indices_u = next(iter(loader_unrolled))

    assert images_u.shape == (4, 3, 256, 256)
    assert targets_u.shape[0] == 4
    assert lengths_u.shape == (4,)
    assert img_indices_u is None
    for i in range(3):
        assert lengths_u[i] >= lengths_u[i + 1]
    for i in range(4):
        assert targets_u[i, 0].item() == Vocabulary.START_IDX
        assert targets_u[i, lengths_u[i] - 1].item() == Vocabulary.END_IDX


if __name__ == "__main__":
    print("Running test_dataset_counts...")
    test_dataset_counts()
    print("Running test_torch_dataset_and_dataloader...")
    test_torch_dataset_and_dataloader()
    print("All dataset tests passed successfully!")

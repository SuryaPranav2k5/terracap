"""Tests for NWPU-Caption dataset loader."""

from image_recognition.dataset import load_dataset_items, load_hf_dataset


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


def test_hf_dataset():
    hf_ds = load_hf_dataset()
    assert "train" in hf_ds
    assert len(hf_ds["train"]) == 31501


if __name__ == "__main__":
    print("Running test_dataset_counts...")
    test_dataset_counts()
    print("Running test_hf_dataset...")
    test_hf_dataset()
    print("All tests passed successfully!")

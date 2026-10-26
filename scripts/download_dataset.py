"""Script to download and organize the NWPU-Caption dataset."""

import os
import tarfile
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download


def setup_dataset(target_dir: str = "dataset") -> None:
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    images_dir = target_path / "images"
    hf_dir = target_path / "hf"

    print("Step 1: Loading dataset via Hugging Face datasets...")
    dataset = load_dataset("KhangTruong/NWPU-Caption")
    print(f"Hugging Face dataset loaded: {dataset}")

    print(f"Step 2: Saving Hugging Face dataset to {hf_dir}...")
    dataset.save_to_disk(str(hf_dir))
    print("Hugging Face dataset successfully saved to disk.")

    print("Step 3: Extracting captions (dataset_nwpu.json)...")
    caption_tar_path = hf_hub_download(
        repo_id="KhangTruong/NWPU-Caption",
        filename="02_NWPU_caption.tar",
        repo_type="dataset",
    )
    with tarfile.open(caption_tar_path, "r") as tar:
        for member in tar.getmembers():
            if member.name.endswith("dataset_nwpu.json"):
                f = tar.extractfile(member)
                if f:
                    with open(target_path / "dataset_nwpu.json", "wb") as out_f:
                        out_f.write(f.read())
                    print(f"Extracted: {target_path / 'dataset_nwpu.json'}")

    print("Step 4: Extracting images to dataset/images/...")
    images_tar_path = hf_hub_download(
        repo_id="KhangTruong/NWPU-Caption",
        filename="02_NWPU_RESISC45.tar",
        repo_type="dataset",
    )
    images_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(images_tar_path, "r") as tar:
        for member in tar.getmembers():
            # Stripping the leading '02_NWPU_RESISC45/' folder prefix
            parts = Path(member.name).parts
            if len(parts) > 1 and parts[0] == "02_NWPU_RESISC45":
                rel_path = Path(*parts[1:])
                dest_path = images_dir / rel_path
                if member.isdir():
                    dest_path.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    f_in = tar.extractfile(member)
                    if f_in:
                        with open(dest_path, "wb") as f_out:
                            f_out.write(f_in.read())

    print(f"All images extracted to {images_dir}")
    print("Dataset setup complete!")


if __name__ == "__main__":
    setup_dataset()

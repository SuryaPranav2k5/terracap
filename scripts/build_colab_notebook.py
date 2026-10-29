"""Generates notebooks/colab_training.ipynb from scripts/colab_standalone.py."""

import json
from pathlib import Path

notebook_dir = Path("notebooks")
notebook_dir.mkdir(exist_ok=True)

with open("scripts/colab_standalone.py", "r", encoding="utf-8") as f:
    script_lines = f.readlines()

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Remote Sensing Image Captioning (Google Colab Edition)\n",
            "### ConvNeXt-Tiny (ImageNet-12k ft 1k) + LoRA PEFT + Spatial Attention LSTM (Show, Attend & Tell)\n",
            "\n",
            "This notebook trains an end-to-end composite image captioning model on the NWPU-Caption dataset in Google Colab.\n",
            "\n",
            "#### Architecture & Key Features:\n",
            "- **Encoder**: `convnext_tiny.in12k_ft_in1k` via `timm` with Low-Rank Adaptation (`peft` LoRA on MLP fc1/fc2 — 1.06M trainable parameters).\n",
            "- **1-Pass Batched Feature Expansion**: Visual features computed **only once** per image and expanded across all 5 captions (5x speedup).\n",
            "- **Decoder**: Bahdanau spatial attention over 8x8 feature grid + LSTM with Show-Attend-Tell deep output layer.\n",
            "- **Evaluation**: True Beam Search ($k=3$) with length normalization and standard `pycocoevalcap` benchmark metrics (BLEU-1..4, CIDEr unscaled, ROUGE-L, METEOR).\n",
            "- **Colab Persistence**: Automatically mounts Google Drive (optional) to safeguard checkpoints (`best_model.pt`) and evaluation reports.\n",
            "- **Lean Checkpoints**: Saves only trainable weights (~12MB vs 120MB).\n",
            "\n",
            "> **Hardware Accelerator**: Make sure GPU is enabled under **Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU (or A100)**.\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Environment & Hardware Verification\n",
            "import torch\n",
            "print(f\"PyTorch Version:  {torch.__version__}\")\n",
            "print(f\"CUDA Available:   {torch.cuda.is_available()}\")\n",
            "if torch.cuda.is_available():\n",
            "    print(f\"GPU Accelerator:  {torch.cuda.get_device_name(0)}\")\n",
            "    print(f\"VRAM:             {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB\")\n",
            "else:\n",
            "    print(\"WARNING: No GPU detected! Go to Runtime -> Change runtime type -> Select T4 GPU.\")\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 2. Install Required Dependencies\n",
            "!pip install -q timm peft pycocoevalcap\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. (Optional) Mount Google Drive for Persistent Storage\n",
            "# If mounted, best_model.pt and reports will automatically be backed up to Drive!\n",
            "try:\n",
            "    from google.colab import drive\n",
            "    drive.mount('/content/drive')\n",
            "    print(\"Google Drive mounted successfully at /content/drive\")\n",
            "except Exception as e:\n",
            "    print(f\"Note: Running with local ephemeral storage (/content): {e}\")\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 4. Dataset Setup (Auto-Extraction if dataset.zip is provided)\n",
            "# If you uploaded dataset.zip to /content or have it in Google Drive, this extracts it\n",
            "# directly to high-speed local SSD storage (/content/dataset).\n",
            "import os\n",
            "import zipfile\n",
            "from pathlib import Path\n",
            "\n",
            "zip_sources = [\n",
            "    Path(\"/content/dataset.zip\"),\n",
            "    Path(\"/content/drive/MyDrive/dataset.zip\"),\n",
            "    Path(\"/content/drive/MyDrive/NWPU_Captioning/dataset.zip\"),\n",
            "    Path(\"/content/drive/MyDrive/Image_Recognition/dataset.zip\"),\n",
            "    Path(\"dataset.zip\"),\n",
            "]\n",
            "\n",
            "extracted = False\n",
            "if not (Path(\"/content/dataset/dataset_nwpu.json\").exists() or Path(\"dataset/dataset_nwpu.json\").exists()):\n",
            "    for z in zip_sources:\n",
            "        if z.exists():\n",
            "            print(f\"Found archive {z}. Extracting to /content...\")\n",
            "            with zipfile.ZipFile(z, 'r') as zip_ref:\n",
            "                zip_ref.extractall(\"/content\" if Path(\"/content\").exists() else \".\")\n",
            "            print(\"Extraction complete!\")\n",
            "            extracted = True\n",
            "            break\n",
            "    if not extracted:\n",
            "        print(\"Notice: No dataset.zip found. Ensure dataset/ folder or dataset.zip is placed in /content or Google Drive.\")\n",
            "else:\n",
            "    print(\"Dataset already extracted and ready!\")\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": script_lines,
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 5. Display Word-by-Word Spatial Attention Heatmaps Inline\n",
            "from IPython.display import Image as ColabImage, display\n",
            "from pathlib import Path\n",
            "\n",
            "vis_dir = Path(\"/content/reports/heatmaps\" if Path(\"/content/reports/heatmaps\").exists() else \"reports/heatmaps\")\n",
            "heatmaps = sorted(list(vis_dir.glob(\"*.png\")))\n",
            "print(f\"Generated {len(heatmaps)} attention heatmap visualizations:\")\n",
            "for hm in heatmaps:\n",
            "    print(f\"\\nHeatmap: {hm.name}\")\n",
            "    display(ColabImage(filename=str(hm)))\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 6. Download Trained Checkpoint and Evaluation Reports (Optional)\n",
            "import shutil\n",
            "from pathlib import Path\n",
            "\n",
            "# Create a single zip archive containing best_model.pt and test reports\n",
            "archive_path = Path(\"/content/nwpu_model_artifacts\")\n",
            "output_root = Path(\"/content\" if Path(\"/content\").exists() else \".\")\n",
            "shutil.make_archive(str(archive_path), \"zip\", root_dir=str(output_root), base_dir=\"reports\")\n",
            "print(f\"Created artifacts bundle: {archive_path}.zip\")\n",
            "\n",
            "# Uncomment to trigger direct browser download in Google Colab:\n",
            "# from google.colab import files\n",
            "# files.download(str(output_root / \"checkpoints\" / \"best_model.pt\"))\n",
            "# files.download(f\"{archive_path}.zip\")\n",
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Summary of Artifacts Produced\n",
            "1. **`checkpoints/best_model.pt`**: Lean checkpoint (~12MB) containing fine-tuned LoRA adapter and spatial attention LSTM weights.\n",
            "2. **`reports/test_metrics.json`**: Official test split benchmark metrics (BLEU-1..4, CIDEr unscaled, ROUGE-L, METEOR).\n",
            "3. **`reports/heatmaps/`**: Word-by-word spatial attention heatmap overlays on satellite imagery.\n",
            "4. **Google Drive Mirror**: If Drive was mounted, all files are also preserved at `My Drive/NWPU_Captioning/`.\n",
        ],
    },
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "toc_visible": True,
        },
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

nb_out = notebook_dir / "colab_training.ipynb"
with open(nb_out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Generated {nb_out.resolve()}")


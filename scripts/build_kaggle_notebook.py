"""Generates notebooks/kaggle_training.ipynb from scripts/kaggle_standalone.py."""

import json
from pathlib import Path

notebook_dir = Path("notebooks")
notebook_dir.mkdir(exist_ok=True)

with open("scripts/kaggle_standalone.py", "r", encoding="utf-8") as f:
    script_lines = f.readlines()

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Remote Sensing Image Captioning\n",
            "### ConvNeXt-Tiny (ImageNet-12k ft 1k) + LoRA PEFT + Spatial Attention LSTM (Show, Attend & Tell)\n",
            "\n",
            "This notebook trains an end-to-end composite image captioning model on the cleaned NWPU-Caption dataset.\n",
            "- **Image Encoder**: `convnext_tiny.in12k_ft_in1k` via `timm` with Low-Rank Adaptation (`peft` LoRA on MLP fc1/fc2).\n",
            "- **Decoder**: Bahdanau spatial attention over 8x8 feature grid + LSTM cell with doubly-stochastic attention penalty.\n",
            "- **Evaluation**: Standard metrics (BLEU-1, BLEU-2, BLEU-3, BLEU-4, ROUGE-L) and word-by-word spatial attention heatmaps.\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# 1. Environment & Package Setup\n",
            "!pip install -q timm peft pycocoevalcap\n",
            "import torch\n",
            'print("PyTorch version:", torch.__version__)\n',
            'print("CUDA available:", torch.cuda.is_available())\n',
            "if torch.cuda.is_available():\n",
            '    print("GPU:", torch.cuda.get_device_name(0))\n',
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
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "### Artifacts & Checkpoints Produced\n",
            "1. `checkpoints/best_model.pt`: Weights achieving the lowest validation loss.\n",
            "2. `reports/test_metrics.json`: Quantitative benchmark metrics on test split.\n",
            "3. `reports/heatmaps/`: Word-by-word spatial attention heatmap overlays on satellite imagery.\n",
        ],
    },
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

nb_out = notebook_dir / "kaggle_training.ipynb"
with open(nb_out, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print(f"Generated {nb_out.resolve()}")

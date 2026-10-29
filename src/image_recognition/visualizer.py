"""Visualization utilities for spatial attention heatmaps."""

import math
from pathlib import Path
from typing import List, Optional, Union
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F


def visualize_attention(
    image: Union[Image.Image, torch.Tensor],
    tokens: List[str],
    attention_maps: torch.Tensor,
    save_path: Optional[Union[str, Path]] = None,
    cols: int = 4,
    cmap: str = "jet",
    alpha: float = 0.55,
) -> plt.Figure:
    """Visualizes spatial attention heatmaps overlaid on the input image for each generated word.

    Args:
        image: Original RGB PIL image or normalized Tensor of shape (3, H, W).
        tokens: List of generated word tokens (length T).
        attention_maps: Tensor of shape (T, H_feat, W_feat) with values summing to 1 per map.
        save_path: Optional filepath to save the resulting visualization.
        cols: Number of subplot columns.
        cmap: Colormap for attention overlay ('jet', 'plasma', 'viridis').
        alpha: Blending transparency of the attention overlay.

    Returns:
        matplotlib.figure.Figure instance.
    """
    if isinstance(image, torch.Tensor):
        # Denormalize standard ImageNet normalization
        img_tensor = image.clone().detach().cpu()
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = img_tensor * std + mean
        img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
        pil_img = Image.fromarray((img_tensor.permute(1, 2, 0).numpy() * 255).astype("uint8"))
    else:
        pil_img = image.convert("RGB")

    width, height = pil_img.size
    num_words = len(tokens)
    total_panels = num_words + 1  # 1 panel for original image + 1 per word

    rows = math.ceil(total_panels / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.8, rows * 3.8))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    # Panel 0: Original image
    axes_flat[0].imshow(pil_img)
    axes_flat[0].set_title("Original Image", fontsize=11, fontweight="bold")
    axes_flat[0].axis("off")

    # Word-by-word attention heatmaps
    if num_words > 0 and attention_maps.dim() == 3:
        # Upsample attention maps (T, 1, H_feat, W_feat) -> (T, height, width)
        upsampled_att = F.interpolate(
            attention_maps.unsqueeze(1).float(),
            size=(height, width),
            mode="bicubic",
            align_corners=False,
        ).squeeze(1).numpy()

        for idx, word in enumerate(tokens):
            ax = axes_flat[idx + 1]
            ax.imshow(pil_img)
            att_map = upsampled_att[idx]
            # Normalize map to [0, 1] for contrast
            map_min, map_max = att_map.min(), att_map.max()
            if map_max > map_min:
                att_map = (att_map - map_min) / (map_max - map_min)

            ax.imshow(att_map, cmap=cmap, alpha=alpha)
            ax.set_title(f"Word: '{word}'", fontsize=11, fontweight="bold")
            ax.axis("off")

    # Hide any unused subplots
    for j in range(total_panels, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()

    if save_path:
        out_file = Path(save_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_file, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return fig

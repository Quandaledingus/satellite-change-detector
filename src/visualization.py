"""Turning analysis arrays into pictures a human can look at."""

import numpy as np
from matplotlib import colormaps


def difference_heatmap(difference: np.ndarray, cmap: str = "inferno") -> np.ndarray:
    """Color a (H, W) difference map with a matplotlib colormap.

    Returns a uint8 RGB array of shape (H, W, 3). The color scale is fixed to
    0-1 so heatmaps from different image pairs can be compared fairly.
    """
    colored = colormaps[cmap](np.clip(difference, 0.0, 1.0))
    return (colored[..., :3] * 255).astype(np.uint8)


def mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[float, float, float] = (1.0, 0.0, 1.0),
    alpha: float = 0.6,
) -> np.ndarray:
    """Paint changed pixels on top of an RGB image.

    Args:
        image: float RGB array (H, W, 3) in [0, 1].
        mask: boolean array (H, W); True pixels get tinted.
        color: tint color as RGB floats. Magenta by default because it
            rarely shows up in natural landscapes.
        alpha: how strongly to tint (0 = invisible, 1 = solid color).

    Returns:
        uint8 RGB array (H, W, 3).
    """
    out = image.copy()
    out[mask] = (1 - alpha) * out[mask] + alpha * np.array(color)
    return (out * 255).astype(np.uint8)

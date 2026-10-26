"""Basic pixel-wise change detection.

The simplest possible approach: subtract the two images, look at how big the
difference is at each pixel, and call it "changed" if the difference is above
a threshold. This is fast and easy to understand, but it reacts to *any*
difference, including lighting, haze, seasons, and misalignment.
"""

import numpy as np


def _check_unit_float_image(image: np.ndarray, label: str) -> None:
    if not np.issubdtype(image.dtype, np.floating):
        raise TypeError(
            f"{label} must be a float array scaled to [0, 1], got dtype {image.dtype}. "
            "Integer images (like uint8) wrap around when subtracted."
        )
    if not np.all(np.isfinite(image)):
        raise ValueError(f"{label} contains NaN or infinite values.")
    if image.min() < 0.0 or image.max() > 1.0:
        raise ValueError(f"{label} values must be between 0 and 1.")


def absolute_difference(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Per-pixel absolute difference, averaged over color channels.

    Args:
        before: float array (H, W) or (H, W, C) with values in [0, 1].
        after: float array with the same shape as `before`.

    Returns:
        Float array of shape (H, W) with values in [0, 1]. 0 means the pixel
        is identical in both images; 1 means it went from black to white
        (or the reverse) in every channel.
    """
    _check_unit_float_image(before, "before")
    _check_unit_float_image(after, "after")
    if before.shape != after.shape:
        raise ValueError(f"Shape mismatch: {before.shape} vs {after.shape}")

    diff = np.abs(after - before)
    if diff.ndim == 3:
        diff = diff.mean(axis=2)
    return diff


def change_mask(difference: np.ndarray, threshold: float) -> np.ndarray:
    """Turn a difference map into a True/False "changed" mask.

    A pixel counts as changed only if its difference is strictly greater
    than `threshold`.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be between 0 and 1, got {threshold}")
    return difference > threshold


def changed_fraction(mask: np.ndarray) -> float:
    """Fraction of pixels in `mask` that are True (0.0 to 1.0)."""
    if mask.size == 0:
        raise ValueError("Mask is empty.")
    return float(np.count_nonzero(mask) / mask.size)

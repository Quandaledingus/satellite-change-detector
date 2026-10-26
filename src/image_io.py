"""Loading and validating ordinary images (PNG / JPEG).

Phase 1 only handles plain RGB pictures. Georeferenced rasters (GeoTIFF,
Sentinel-2 bands) need Rasterio and come in Phase 2.
"""

import logging
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

# Pillow "modes" describe how pixels are stored. These are all 8 bits per
# channel, so dividing by 255 maps them cleanly onto 0.0-1.0.
SUPPORTED_MODES = {"RGB", "RGBA", "L", "P"}


class ImageValidationError(ValueError):
    """Raised when an image can't be safely used for comparison."""


def load_rgb_image(source: str | Path | BinaryIO) -> np.ndarray:
    """Load an image as a float array of shape (height, width, 3) in [0, 1].

    `source` can be a file path or an open file-like object (for example a
    Streamlit upload).

    Raises:
        ImageValidationError: if the file type is unsupported, the file is
            not a readable image, or its pixel format isn't supported.
    """
    name = str(source) if isinstance(source, (str, Path)) else getattr(source, "name", "")
    suffix = Path(name).suffix.lower()
    if suffix and suffix not in SUPPORTED_EXTENSIONS:
        raise ImageValidationError(
            f"Unsupported file type '{suffix}'. Use one of: {sorted(SUPPORTED_EXTENSIONS)}."
        )

    try:
        with Image.open(source) as img:
            img.load()
            mode = img.mode
            if mode not in SUPPORTED_MODES:
                raise ImageValidationError(
                    f"Unsupported pixel format '{mode}'. Only 8-bit RGB, RGBA, "
                    "grayscale or palette images are supported right now."
                )
            if mode == "RGBA" and img.getextrema()[3][0] < 255:
                logger.warning(
                    "%s has transparent pixels; transparency is ignored for now.", name or "image"
                )
            rgb = img.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        label = name or "image"
        raise ImageValidationError(f"Could not read '{label}' as an image: {exc}") from exc

    return np.asarray(rgb, dtype=np.float32) / 255.0


def validate_same_shape(before: np.ndarray, after: np.ndarray) -> None:
    """Make sure two images can be compared pixel-by-pixel.

    Raises:
        ImageValidationError: if the arrays have different shapes.
    """
    if before.shape != after.shape:
        raise ImageValidationError(
            f"Images must be the same size to compare them. "
            f"Before is {before.shape[1]}x{before.shape[0]} pixels, "
            f"after is {after.shape[1]}x{after.shape[0]} pixels."
        )

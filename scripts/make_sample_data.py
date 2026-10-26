"""Generate a synthetic before/after image pair with a known ground-truth mask.

This is NOT real satellite data. It's a fake landscape made of four land-cover
classes (vegetation, bare soil, water, buildings). Because we create the
changes ourselves, we know exactly which pixels really changed, which is
useful for testing the algorithm.

The "after" image also gets a small brightness shift and random noise so it
behaves a bit like a second satellite pass under different lighting.

Usage:
    python scripts/make_sample_data.py
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

VEGETATION, SOIL, WATER, BUILT = 0, 1, 2, 3

# Rough RGB colors for each land-cover class (floats in 0-1).
CLASS_COLORS = np.array(
    [
        [0.16, 0.38, 0.14],  # vegetation
        [0.52, 0.42, 0.30],  # bare soil
        [0.10, 0.20, 0.38],  # water
        [0.62, 0.62, 0.64],  # buildings / concrete
    ]
)


def smooth_noise(rng: np.random.Generator, size: int, cells: int) -> np.ndarray:
    """Random blobs: a small random grid scaled up smoothly to size x size."""
    coarse = (rng.random((cells, cells)) * 255).astype(np.uint8)
    big = Image.fromarray(coarse).resize((size, size), Image.Resampling.BICUBIC)
    return np.asarray(big, dtype=np.float64) / 255.0


def make_class_maps(rng: np.random.Generator, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Build 'before' and 'after' land-cover maps (2D arrays of class ids)."""
    before = np.full((size, size), VEGETATION, dtype=np.uint8)
    before[smooth_noise(rng, size, 8) > 0.72] = SOIL

    # A winding river
    rows, cols = np.mgrid[0:size, 0:size]
    river_center = size * 0.55 + size * 0.08 * np.sin(cols / size * 2 * np.pi)
    before[np.abs(rows - river_center) < size * 0.03] = WATER

    after = before.copy()

    # Change 1: a patch of forest is cleared (vegetation -> soil)
    s = size / 256
    clearing = np.zeros_like(before, dtype=bool)
    clearing[int(30 * s) : int(95 * s), int(140 * s) : int(215 * s)] = True
    after[clearing & (before == VEGETATION)] = SOIL

    # Change 2: a new building complex appears on land (not in the river)
    building = np.zeros_like(before, dtype=bool)
    building[int(190 * s) : int(225 * s), int(35 * s) : int(85 * s)] = True
    after[building & (before != WATER)] = BUILT

    return before, after


def render(classes: np.ndarray, rng: np.random.Generator, brightness: float) -> np.ndarray:
    """Turn a class map into an RGB image with texture, lighting and noise."""
    size = classes.shape[0]
    image = CLASS_COLORS[classes]
    texture = 0.85 + 0.3 * smooth_noise(rng, size, 32)
    image = image * texture[..., None] * brightness
    image = image + rng.normal(0.0, 0.015, image.shape)
    return np.clip(image, 0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("data/sample"))
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    before_classes, after_classes = make_class_maps(rng, args.size)
    before = render(before_classes, rng, brightness=1.0)
    after = render(after_classes, rng, brightness=1.08)
    truth = before_classes != after_classes

    args.out.mkdir(parents=True, exist_ok=True)
    Image.fromarray((before * 255).round().astype(np.uint8)).save(args.out / "before.png")
    Image.fromarray((after * 255).round().astype(np.uint8)).save(args.out / "after.png")
    Image.fromarray(truth.astype(np.uint8) * 255).save(args.out / "truth_mask.png")
    print(f"Wrote sample pair to {args.out}/ ({truth.mean():.1%} of pixels truly changed)")


if __name__ == "__main__":
    main()

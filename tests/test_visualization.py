import numpy as np

from src.visualization import difference_heatmap, mask_overlay


def test_heatmap_shape_and_dtype():
    heat = difference_heatmap(np.linspace(0, 1, 12).reshape(3, 4))

    assert heat.shape == (3, 4, 3)
    assert heat.dtype == np.uint8


def test_overlay_only_changes_masked_pixels():
    image = np.full((2, 2, 3), 0.5)
    mask = np.array([[True, False], [False, False]])

    out = mask_overlay(image, mask, color=(1.0, 0.0, 0.0), alpha=1.0)

    assert out[0, 0].tolist() == [255, 0, 0]
    assert out[1, 1].tolist() == [127, 127, 127]
    assert image[0, 0, 0] == 0.5  # original is not modified

import numpy as np
import pytest
from PIL import Image

from src.image_io import ImageValidationError, load_rgb_image, validate_same_shape


def save_png(path, array, mode=None):
    Image.fromarray(array, mode=mode).save(path)
    return path


def test_loads_rgb_png_scaled_to_unit_range(tmp_path):
    pixels = np.array([[[0, 128, 255], [255, 255, 255]]], dtype=np.uint8)
    path = save_png(tmp_path / "img.png", pixels)

    image = load_rgb_image(path)

    assert image.shape == (1, 2, 3)
    assert image.dtype == np.float32
    np.testing.assert_allclose(image[0, 0], [0.0, 128 / 255, 1.0])


def test_grayscale_becomes_three_channels(tmp_path):
    path = save_png(tmp_path / "gray.png", np.full((4, 5), 200, dtype=np.uint8))

    image = load_rgb_image(path)

    assert image.shape == (4, 5, 3)
    np.testing.assert_allclose(image, 200 / 255)


def test_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("hello")

    with pytest.raises(ImageValidationError, match="Unsupported file type"):
        load_rgb_image(path)


def test_rejects_corrupt_file(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"this is not really a png")

    with pytest.raises(ImageValidationError, match="Could not read"):
        load_rgb_image(path)


def test_rejects_16_bit_images(tmp_path):
    path = tmp_path / "deep.png"
    Image.fromarray(np.full((3, 3), 40000, dtype=np.uint16)).save(path)

    with pytest.raises(ImageValidationError, match="Unsupported pixel format"):
        load_rgb_image(path)


def test_accepts_file_like_objects(tmp_path):
    path = save_png(tmp_path / "img.png", np.zeros((2, 2, 3), dtype=np.uint8))

    with open(path, "rb") as f:
        image = load_rgb_image(f)

    assert image.shape == (2, 2, 3)


def test_same_shape_passes():
    validate_same_shape(np.zeros((3, 4, 3)), np.ones((3, 4, 3)))


def test_different_shape_raises():
    with pytest.raises(ImageValidationError, match="same size"):
        validate_same_shape(np.zeros((3, 4, 3)), np.zeros((4, 3, 3)))

import numpy as np
import pytest

from src.change_detection import absolute_difference, change_mask, changed_fraction


def test_identical_images_have_zero_difference():
    image = np.random.default_rng(0).random((5, 5, 3))

    diff = absolute_difference(image, image)

    assert diff.shape == (5, 5)
    assert np.all(diff == 0)


def test_difference_is_averaged_over_channels():
    before = np.zeros((1, 1, 3))
    after = np.array([[[0.3, 0.6, 0.9]]])

    diff = absolute_difference(before, after)

    assert diff[0, 0] == pytest.approx(0.6)


def test_difference_is_symmetric():
    rng = np.random.default_rng(1)
    a, b = rng.random((4, 4, 3)), rng.random((4, 4, 3))

    np.testing.assert_allclose(absolute_difference(a, b), absolute_difference(b, a))


def test_works_on_single_channel_images():
    diff = absolute_difference(np.zeros((2, 2)), np.full((2, 2), 0.25))

    np.testing.assert_allclose(diff, 0.25)


def test_rejects_integer_images():
    # uint8 math wraps around: np.uint8(10) - np.uint8(20) == 246, not -10.
    a = np.full((2, 2, 3), 10, dtype=np.uint8)
    b = np.full((2, 2, 3), 20, dtype=np.uint8)

    with pytest.raises(TypeError, match="float"):
        absolute_difference(a, b)


def test_rejects_values_outside_unit_range():
    with pytest.raises(ValueError, match="between 0 and 1"):
        absolute_difference(np.zeros((2, 2)), np.full((2, 2), 255.0))


def test_rejects_nan():
    bad = np.zeros((2, 2))
    bad[0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN"):
        absolute_difference(np.zeros((2, 2)), bad)


def test_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="Shape mismatch"):
        absolute_difference(np.zeros((2, 2)), np.zeros((3, 3)))


def test_threshold_is_strictly_greater_than():
    diff = np.array([0.1, 0.2, 0.3])

    mask = change_mask(diff, threshold=0.2)

    assert mask.tolist() == [False, False, True]


def test_threshold_zero_flags_any_change():
    assert change_mask(np.array([0.0, 0.001]), threshold=0.0).tolist() == [False, True]


@pytest.mark.parametrize("bad_threshold", [-0.1, 1.5])
def test_threshold_out_of_range_raises(bad_threshold):
    with pytest.raises(ValueError):
        change_mask(np.zeros(3), bad_threshold)


def test_changed_fraction():
    mask = np.array([[True, False], [False, False]])

    assert changed_fraction(mask) == 0.25


def test_changed_fraction_of_empty_mask_raises():
    with pytest.raises(ValueError):
        changed_fraction(np.zeros((0, 0), dtype=bool))

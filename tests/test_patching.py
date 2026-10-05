import numpy as np

from sr_datasets.patching import extract_aligned_patches


def test_aligned_patch_coordinates_and_values() -> None:
    hr = np.arange(3 * 8 * 8, dtype=np.float32).reshape(3, 8, 8)
    lr = hr[:, ::2, ::2]
    patches = list(extract_aligned_patches(hr, lr, 4, 4, 2))
    assert patches[1][2] == (0, 4)
    assert np.array_equal(patches[0][0][:, ::2, ::2], patches[0][1])


def test_odd_tile_dimension_is_allowed() -> None:
    hr = np.ones((3, 7, 9), dtype=np.float32)
    lr = np.ones((3, 3, 4), dtype=np.float32)
    patches = list(extract_aligned_patches(hr, lr, 4, 4, 2))
    assert len(patches) == 2

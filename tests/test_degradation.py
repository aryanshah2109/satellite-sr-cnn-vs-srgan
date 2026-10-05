import numpy as np
import pytest
import torch

from sr_datasets.degradation import degrade_image


def test_degradation_shape_range_and_seed() -> None:
    hr = np.linspace(0.0, 1.0, 3 * 8 * 12, dtype=np.float32).reshape(3, 8, 12)
    config = {"noise_sigma_range": [0.01, 0.01], "jpeg_quality_range": None}
    first = degrade_image(hr, 4, config, torch.Generator().manual_seed(7))
    second = degrade_image(hr, 4, config, torch.Generator().manual_seed(7))
    assert first.shape == (3, 2, 3)
    assert np.array_equal(first, second)
    assert 0.0 <= float(first.min()) <= float(first.max()) <= 1.0


def test_degradation_rejects_invalid_size() -> None:
    with pytest.raises(ValueError, match="divisible"):
        degrade_image(np.zeros((3, 7, 8), dtype=np.float32), 4, {})

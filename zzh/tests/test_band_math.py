# tests/test_band_math.py
import numpy as np
import pytest
from image_processing.core.band_math import ndvi, ndwi, custom_expression


def test_ndvi_range():
    nir = np.array([[0.8, 0.9]], dtype=np.float32)
    red = np.array([[0.1, 0.2]], dtype=np.float32)
    res = ndvi(nir, red)
    assert (res <= 1).all() and (res >= -1).all()


def test_custom_expr():
    b1 = np.ones((2, 2), dtype=np.float32)
    b2 = np.zeros((2, 2), dtype=np.float32)
    res = custom_expression("(B1 + B2) / 2", b1, b2)
    assert (res == 0.5).all()


def test_invalid_expr():
    with pytest.raises(ValueError):
        custom_expression("__import__('os').system('rm -rf /')", np.ones((2, 2)))

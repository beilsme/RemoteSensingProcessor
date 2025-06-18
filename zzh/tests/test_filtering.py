# tests/test_filtering.py
import numpy as np
from image_processing.core.filtering.smoothing import (
    smooth_mean, smooth_gaussian, smooth_median
)
from image_processing.core.filtering.sharpening import (
    sharpen_unsharp, sharpen_laplacian
)
from image_processing.core.filtering.edge_detection import (
    edge_sobel, edge_canny
)


def test_smoothing_variance(sample_array):
    """平滑应降低方差"""
    for func in (smooth_mean, smooth_gaussian, smooth_median):
        assert func(sample_array).var() < sample_array.var()


def test_sharpening_variance(sample_array):
    """锐化通常提升方差"""
    for func in (sharpen_unsharp, sharpen_laplacian):
        assert func(sample_array).var() > sample_array.var()


def test_edge_output_shape(sample_array):
    """边缘检测输出单波段"""
    sobel = edge_sobel(sample_array)
    canny = edge_canny(sample_array, 1.0)
    assert sobel.ndim == 2
    assert canny.ndim == 2
    assert sobel.shape == canny.shape

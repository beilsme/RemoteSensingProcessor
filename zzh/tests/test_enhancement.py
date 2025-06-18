# tests/test_enhancement.py
import numpy as np
from image_processing.core.enhancement.image_stretching import (
    stretch_linear, stretch_percent
)
from image_processing.core.enhancement.equalization import hist_equalize


def test_linear_identity(sample_array):
    """src_min / src_max = 数据真实范围时，最小应映射到 0，最大映射到 1"""
    arr = sample_array
    out = stretch_linear(arr, src_min=arr.min(), src_max=arr.max())
    assert np.isclose(out.min(), 0.0)
    assert np.isclose(out.max(), 1.0)


def test_percent_range(sample_array):
    """百分比拉伸后，输出仍在 0‒1 区间"""
    out = stretch_percent(sample_array, 2, 98)
    assert 0.0 <= out.min() <= out.max() <= 1.0


def test_equalization_shape(sample_array):
    """均衡化不改变形状与波段数"""
    out = hist_equalize(sample_array)
    assert out.shape == sample_array.shape

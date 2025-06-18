# -*- coding: utf-8 -*-
"""
RemoteSensingProcessor – Array Utilities
----------------------------------------
常用数组操作：nan 处理、归一化等

Created : 2025-06-17
Author  : 张子涵
Version : 0.1.0
"""
import numpy as np

__all__ = ["nan_to_num", "normalize_01"]


def nan_to_num(arr: np.ndarray, fill: float = 0.0) -> np.ndarray:
    """把 nan 填为指定值"""
    out = arr.copy()
    out[np.isnan(out)] = fill
    return out


def normalize_01(arr: np.ndarray) -> np.ndarray:
    """简单 0-1 归一化"""
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    if mx - mn == 0:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)

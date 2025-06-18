# -*- coding: utf-8 -*-
"""
RemoteSensingProcessor – Raster I/O Utilities
--------------------------------------------
封装 rasterio 读写，简化主流程。

Created : 2025-06-17
Author  : 张子涵
Version : 0.1.0
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import rasterio

__all__ = ["read_raster", "write_raster", "copy_profile"]


def read_raster(path: str | Path) -> tuple[np.ndarray, dict]:
    """返回 (array, profile)；array shape 为 (bands, rows, cols)"""
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        profile = src.profile
    return data, profile


def write_raster(path: str | Path, array: np.ndarray,
                 profile: dict, dtype=rasterio.float32) -> None:
    profile = profile.copy()
    profile.update(count=array.shape[0], dtype=dtype)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(array.astype(dtype))


def copy_profile(profile: dict, *, count: int | None = None,
                 dtype=rasterio.float32) -> dict:
    """深复制 profile 并可修改波段数 / 数据类型"""
    new_p = profile.copy()
    if count is not None:
        new_p["count"] = count
    new_p["dtype"] = dtype
    return new_p

# tests/conftest.py
import numpy as np
import pytest
from pathlib import Path
import rasterio
from rasterio.transform import from_origin
import os


@pytest.fixture
def sample_array():
    """返回 3×4×4 随机影像（3 波段）"""
    np.random.seed(0)
    return (np.random.rand(3, 4, 4) * 1000).astype(np.float32)


@pytest.fixture
def tmp_tif(tmp_path, sample_array):
    """在临时目录写入内存 GeoTIFF，供 API 测试"""
    path = Path(tmp_path) / "test.tif"
    height, width = sample_array.shape[1:]
    profile = {
        "driver": "GTiff",
        "count": sample_array.shape[0],
        "height": height,
        "width": width,
        "dtype": "float32",
        # 某些测试环境中 PROJ 数据库版本较旧，解析 EPSG 代码会失败，
        # 因此测试影像不设置坐标参考，以避免依赖外部投影库。
        "transform": from_origin(0, 0, 1, 1),
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(sample_array)
    return path


@pytest.fixture(scope="session")
def real_image():
    """如果环境变量 ``RSP_TEST_IMAGE`` 指定了影像路径，则返回该路径。"""
    img_path = os.environ.get("RSP_TEST_IMAGE")
    if not img_path:
        pytest.skip("RSP_TEST_IMAGE 未设置，跳过真实影像测试")
    p = Path(img_path)
    if not p.exists():
        pytest.skip(f"测试影像不存在: {img_path}")
    return p
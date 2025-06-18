# tests/test_api_pipeline.py
import rasterio
from image_processing.api.enhancement_api import percent_stretch_raster
from image_processing.api.band_math_api import ndvi_raster


def test_percent_stretch_pipeline(tmp_tif, tmp_path):
    out = tmp_path / "stretch.tif"
    percent_stretch_raster(tmp_tif, out, 1, 99)
    with rasterio.open(out) as ds:
        data = ds.read()
        assert data.min() >= 0 and data.max() <= 1


def test_ndvi_pipeline(tmp_tif, tmp_path):
    out = tmp_path / "ndvi.tif"
    ndvi_raster(tmp_tif, out, nir_band=1, red_band=2)  # 用随机影像模拟
    with rasterio.open(out) as ds:
        assert ds.count == 1

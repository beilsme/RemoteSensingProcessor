import rasterio
from image_processing.api.enhancement_api import percent_stretch_raster
from image_processing.api.band_math_api import ndvi_raster


def test_real_percent_stretch(real_image, tmp_path):
    out = tmp_path / "stretch.tif"
    percent_stretch_raster(real_image, out, 1, 99)
    with rasterio.open(out) as ds:
        data = ds.read()
        assert data.min() >= 0 and data.max() <= 1


def test_real_ndvi(real_image, tmp_path):
    out = tmp_path / "ndvi.tif"
    ndvi_raster(real_image, out)
    with rasterio.open(out) as ds:
        assert ds.count == 1
# -*- coding: utf-8 -*-
"""
API – Band Math
---------------

Created : 2025-06-17
Author  : 张子涵
Version : 0.1.0
"""
from pathlib import Path
from ..utils.raster_io import read_raster, write_raster, copy_profile
from ..utils.logging import init_logger
from ..core.band_math import ndvi, ndwi, custom_expression

log = init_logger("BandMathAPI")


def ndvi_raster(in_raster: str | Path, out_raster: str | Path,
                nir_band: int = 4, red_band: int = 3):
    data, prof = read_raster(in_raster)
    nir, red = data[nir_band - 1], data[red_band - 1]
    log.info(f"NDVI (nir={nir_band}, red={red_band})")
    out = ndvi(nir, red)
    write_raster(out_raster, out[None, ...], copy_profile(prof, count=1))


def ndwi_raster(in_raster: str | Path, out_raster: str | Path,
                nir_band: int = 4, green_band: int = 2):
    data, prof = read_raster(in_raster)
    nir, green = data[nir_band - 1], data[green_band - 1]
    log.info(f"NDWI (nir={nir_band}, green={green_band})")
    out = ndwi(nir, green)
    write_raster(out_raster, out[None, ...], copy_profile(prof, count=1))


def expr_raster(in_raster: str | Path, out_raster: str | Path,
                expr: str, band_mapping: list[int]):
    data, prof = read_raster(in_raster)
    bands = [data[i - 1] for i in band_mapping]
    log.info(f"Custom expr '{expr}' with bands {band_mapping}")
    out = custom_expression(expr, *bands)
    write_raster(out_raster, out[None, ...], copy_profile(prof, count=1))

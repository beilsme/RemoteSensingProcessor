# -*- coding: utf-8 -*-
"""
API – Image Enhancement
-----------------------

Created : 2025-06-17
Author  : 张子涵
Version : 0.1.0
"""
from pathlib import Path
from ..utils.raster_io import read_raster, write_raster, copy_profile
from ..utils.logging import init_logger
from ..core.enhancement.image_stretching import stretch_linear, stretch_percent
from ..core.enhancement.equalization import hist_equalize

log = init_logger("EnhancementAPI")


def linear_stretch_raster(in_raster: str | Path, out_raster: str | Path,
                          src_min=None, src_max=None):
    """Apply linear stretch and write result to a new raster."""
    img, prof = read_raster(in_raster)
    log.info("Linear stretching...")
    out = stretch_linear(img, src_min, src_max)
    write_raster(out_raster, out, copy_profile(prof, dtype="float32"))


def percent_stretch_raster(in_raster: str | Path, out_raster: str | Path,
                           low=2.0, high=98.0):
    """Percent-based stretch using low/high percentiles."""
    img, prof = read_raster(in_raster)
    log.info("Percent stretching %.1f-%.1f" % (low, high))
    out = stretch_percent(img, low, high)
    write_raster(out_raster, out, copy_profile(prof, dtype="float32"))


def hist_equalize_raster(in_raster: str | Path, out_raster: str | Path):
    """Histogram equalization on all bands."""
    img, prof = read_raster(in_raster)
    log.info("Histogram equalization")
    out = hist_equalize(img)
    write_raster(out_raster, out, copy_profile(prof, dtype="float32"))

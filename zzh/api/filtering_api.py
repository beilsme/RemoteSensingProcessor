# -*- coding: utf-8 -*-
"""
API – Filtering
---------------

Created : 2025-06-17
Author  : 张子涵
Version : 0.1.0
"""
from pathlib import Path
from ..utils.raster_io import read_raster, write_raster, copy_profile
from ..utils.logging import init_logger
from ..core.filtering.smoothing import smooth_mean, smooth_gaussian, smooth_median
from ..core.filtering.sharpening import sharpen_unsharp, sharpen_laplacian
from ..core.filtering.edge_detection import edge_sobel, edge_canny

log = init_logger("FilteringAPI")


def mean_filter_raster(in_raster: str | Path, out_raster: str | Path, size=3):
    """Apply mean filter with given kernel size."""
    arr, prof = read_raster(in_raster)
    log.info(f"Mean filter size={size}")
    out = smooth_mean(arr, size)
    write_raster(out_raster, out, copy_profile(prof))


def gaussian_filter_raster(in_raster: str | Path, out_raster: str | Path, sigma=1.0):
    """Gaussian smoothing with standard deviation ``sigma``."""
    arr, prof = read_raster(in_raster)
    log.info(f"Gaussian filter σ={sigma}")
    out = smooth_gaussian(arr, sigma)
    write_raster(out_raster, out, copy_profile(prof))


def median_filter_raster(in_raster: str | Path, out_raster: str | Path, size=3):
    """Median filter with kernel size ``size``."""
    arr, prof = read_raster(in_raster)
    log.info(f"Median filter size={size}")
    out = smooth_median(arr, size)
    write_raster(out_raster, out, copy_profile(prof))


def unsharp_raster(in_raster: str | Path, out_raster: str | Path,
                   radius=1.0, amount=1.0):
    """Unsharp masking on each band."""
    arr, prof = read_raster(in_raster)
    log.info(f"Unsharp mask r={radius}, amount={amount}")
    out = sharpen_unsharp(arr, radius, amount)
    write_raster(out_raster, out, copy_profile(prof))


def laplacian_raster(in_raster: str | Path, out_raster: str | Path, alpha=1.0):
    """Laplacian sharpening."""
    arr, prof = read_raster(in_raster)
    log.info(f"Laplacian sharpen α={alpha}")
    out = sharpen_laplacian(arr, alpha)
    write_raster(out_raster, out, copy_profile(prof))


def sobel_raster(in_raster: str | Path, out_raster: str | Path):
    """Detect edges using Sobel operator."""
    arr, prof = read_raster(in_raster)
    log.info("Sobel edge")
    out = edge_sobel(arr)
    write_raster(out_raster, out[None, ...], copy_profile(prof, count=1))


def canny_raster(in_raster: str | Path, out_raster: str | Path, sigma=1.0):
    """Canny edge detection with Gaussian sigma."""
    arr, prof = read_raster(in_raster)
    log.info(f"Canny edge σ={sigma}")
    out = edge_canny(arr, sigma)
    write_raster(out_raster, out[None, ...], copy_profile(prof, count=1))
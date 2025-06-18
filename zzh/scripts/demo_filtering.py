#!/usr/bin/env python
"""示例：均值滤波"""
from pathlib import Path
import sys
# 允许直接运行该脚本
PKG_ROOT = Path(__file__).resolve().parents[2]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))
from image_processing.api.filtering_api import mean_filter_raster

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: demo_filtering.py <in> <out>")
        sys.exit(1)
    mean_filter_raster(Path(sys.argv[1]), Path(sys.argv[2]))
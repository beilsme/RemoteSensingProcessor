#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
文件: config.py
模块: 根目录
功能: 默认配置示例，用于测试数据跑通全流程
作者: 孟诣楠
版本: v1.1.0
最近更新: 2025-06-19
较上一版改进:
  1. 新增 feature_extraction_params，配置特征提取阶段的输入/输出路径；
  2. 统一 image_display、image_processing 参数键名；
"""
from src.constants import TEST_DATA_DIR, OUTPUT_DIR
import os
import glob

class EngineConfig:
    """配置容器，所有任务参数以子属性形式挂载"""
    pass

config = EngineConfig()

# 1. 文件操作
config.file_operation_params = {
    "input_dir": TEST_DATA_DIR,
    "output_dir": os.path.join(OUTPUT_DIR, "file_operation")
}

# 2. 文件保存
config.file_saver_params = {
    "save_dir": os.path.join(OUTPUT_DIR, "saved_files")
}

# 3. 波段展示
config.image_display_params = {
    "paths": glob.glob(os.path.join(OUTPUT_DIR, "file_operation", "*.npy")),
    "bands": [1, 2, 3],
    "output_dir": os.path.join(OUTPUT_DIR, "display")
}

# 4. 图像处理
config.image_processing_params = {
    "paths": glob.glob(os.path.join(OUTPUT_DIR, "file_operation", "*.npy")),
    "methods": ["equalization", "stretch"],
    "output_dir": os.path.join(OUTPUT_DIR, "processed"),
    "options": {
        "stretch": {"in_range": (2, 98), "out_range": (0, 255)}
    }
}

# 5. 特征提取（新增）
config.feature_extraction_params = {
    "input_dir": os.path.join(OUTPUT_DIR, "processed"),
    "output_dir": os.path.join(OUTPUT_DIR, "features"),
    "visualization_output_dir": os.path.join(OUTPUT_DIR, "feature_plots")
}

# 6. 矢量处理
config.vector_processing_params = {
    "input_paths": [
        os.path.join(TEST_DATA_DIR, "sample.shp"),
    ],
    "output_dir": os.path.join(OUTPUT_DIR, "vector"),
    "operations": [
        {"type": "reproject", "params": {"crs": getattr(config, "target_crs", None)}},
    ],
    "options": {
        "load": None,
        "save": {"driver": "ESRI Shapefile"}
    }
}

# 7. 分类
model = "random_forest"
config.classification_params = {
    "data": {
        "features": os.path.join(OUTPUT_DIR, "processed", "features.npy"),
        "labels":   os.path.join(OUTPUT_DIR, "processed", "labels.npy")
    },
    "pipeline_config": {
        "classifiers": [
            {"name": model, "params": {"random_state": 0}}
        ],
        "compare": False
    },
    "class_map_path": os.path.join(OUTPUT_DIR, "processed", "class_map.npy"),
    "model": model
}

# 8. 精度评估
config.evaluation_params = {
    "class_map_path": os.path.join(OUTPUT_DIR, "processed", "class_map.npy"),
    "roi_mask_path": os.path.join(TEST_DATA_DIR, "mask.npy"),
    "output_dir": os.path.join(OUTPUT_DIR, "evaluation")
}

if __name__ == "__main__":
    # 测试打印
    import json
    print("=== feature_extraction_params ===")
    print(json.dumps(config.feature_extraction_params, indent=2, ensure_ascii=False))

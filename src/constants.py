#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
文件: constants.py
模块: 根目录
功能: 常量定义
作者: 孟诣楠
版本: v1.0.1
创建时间: 2025-06-19
最近更新: 2025-06-20
较上一版改进:
  1. 修改测试数据目录为 data/test_data；
"""
import os

# 项目根目录
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
# 测试数据目录：请在项目根目录下创建 data/test_data 文件夹
TEST_DATA_DIR = os.path.join(BASE_DIR, "data", "test_data")
# 默认结果输出目录
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

# 其他常量
SUPPORTED_TASKS = [
    "file_operation", "file_saver", "image_display",
    "image_processing", "vector_processing",
    "classification", "evaluation"
]

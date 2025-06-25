# -*- coding: utf-8 -*-
"""
文件: data_adapter.py
模块: src.utils.data_adapter
功能: 将 (H, W, D) 特征数组与 (H, W) 标签掩膜转换为监督分类所需的一维样本形式
作者: 孟诣楠
版本: v1.0.0
最近更新: 2025-06-25

功能说明:
    - 支持从遥感影像提取带标签的有效样本对
    - 自动剔除无效像素与无标签区域
    - 输出适用于训练分类器的 (N, D) 与 (N,) 结构
"""

import numpy as np


def extract_labeled_samples(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    提取监督分类所需的有效样本点（features 与 labels）

    参数:
        features: 原始特征数组，形状为 (H, W, D)
        labels:   掩膜标签数组，形状为 (H, W)

    返回:
        X: 特征样本数组，形状为 (N, D)
        y: 对应标签数组，形状为 (N,)
    """
    if features.ndim != 3 or labels.ndim != 2:
        raise ValueError("特征应为 (H, W, D)，标签应为 (H, W)")

    H1, W1, D = features.shape
    H2, W2 = labels.shape
    if H1 != H2 or W1 != W2:
        raise ValueError(f"特征与标签尺寸不一致: features=({H1},{W1}), labels=({H2},{W2})")

    # 展平处理
    X = features.reshape(-1, D)
    y = labels.reshape(-1)

    # 筛选有效样本（label >= 0 且非 NaN）
    valid_mask = (y >= 0) & np.isfinite(y)
    X_valid = X[valid_mask]
    y_valid = y[valid_mask].astype(int)

    if len(X_valid) == 0:
        raise ValueError("未提取到任何有效样本（可能掩膜为空或全为无效值）")

    return X_valid, y_valid


if __name__ == "__main__":
    # 简单测试
    features = np.random.rand(4, 5, 3)
    labels = np.array([
        [0, 1, -1, 2, 0],
        [1, -1, -1, -1, 2],
        [2, 0, 1, -1, 0],
        [-1, -1, 1, 2, 2]
    ])
    X, y = extract_labeled_samples(features, labels)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("unique labels:", np.unique(y))

#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
文件: run_feature_extraction.py
模块: src.processing.feature_extraction.run_feature_extraction
功能: 自动读取单/多文件遥感波段，统一调度光谱指数、纹理、PCA、形态学、
      特征选择、融合与可视化模块；保存所有特征为 .npy；封装为 TaskResult
作者: 孟诣楠
版本: v1.0.4
最近更新: 2025-06-18

更新说明:
    - 封装返回值为 TaskResult，包括 status, message, outputs, logs
    - 引入 src.processing.task_result.TaskResult
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import rasterio
if __package__ is None or __package__ == "":
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").is_dir():
            sys.path.insert(0, str(parent))
            break
from typing import List, Dict, Any

from src.processing.task_result import TaskResult
from src.processing.feature_extraction.indices       import calculate_ndvi, calculate_evi, calculate_msavi, calculate_ndwi, calculate_mndwi, calculate_ndbi, calculate_bsi
from src.processing.feature_extraction.texture       import calculate_glcm_features, calculate_lbp_features, calculate_gabor_features
from src.processing.feature_extraction.pca           import perform_pca
from src.processing.feature_extraction.morphology    import calculate_morphological_features, calculate_filter_responses
from src.processing.feature_extraction.selection     import feature_selection_by_variance, calculate_multi_scale_features
from src.processing.feature_extraction.fusion        import feature_fusion_for_segmentation, prepare_features_for_segmentation, hierarchical_feature_fusion, add_spatial_context
from src.processing.feature_extraction.visualization import visualize_selected_features, visualize_hierarchical_features

_DEFAULT_BAND_NAMES = ['blue', 'green', 'red', 'nir', 'swir']


def auto_map_bands_from_descriptions(descriptions, count):
    names = []
    for desc in descriptions:
        if not desc:
            names.append(None)
            continue
        low = desc.lower()
        if 'blue' in low:
            names.append('blue')
        elif 'green' in low:
            names.append('green')
        elif 'red' in low:
            names.append('red')
        elif 'nir' in low or 'near infrared' in low:
            names.append('nir')
        elif 'swir' in low or 'shortwave' in low:
            names.append('swir')
        else:
            names.append(None)
    if sum(n is not None for n in names) >= 3:
        return [n if n else f'band_{i+1}' for i, n in enumerate(names)]
    return []


def run(input_files: List[str], output_dir: str) -> TaskResult:
    logs: List[str] = []
    outputs: List[str] = []

    try:
        os.makedirs(output_dir, exist_ok=True)
        logs.append(f"创建输出目录: {output_dir}")

        # 波段加载 & 命名
        band_arrays: Dict[str, np.ndarray] = {}
        if len(input_files) == 1 and input_files[0].lower().endswith(('.tif','.tiff')):
            fp = input_files[0]
            with rasterio.open(fp) as ds:
                descriptions = ds.descriptions or ()
                mapped = auto_map_bands_from_descriptions(descriptions, ds.count)
                if mapped:
                    band_names = mapped
                    logs.append(f"根据描述自动映射波段: {band_names}")
                elif ds.count >= len(_DEFAULT_BAND_NAMES):
                    band_names = _DEFAULT_BAND_NAMES + [f'band_{i+1}' for i in range(len(_DEFAULT_BAND_NAMES), ds.count)]
                    logs.append(f"使用默认顺序映射前五波段: {band_names[:5]}")
                else:
                    band_names = [f'band_{i+1}' for i in range(ds.count)]
                    logs.append(f"使用通用命名: {band_names}")
                for idx, name in enumerate(band_names, start=1):
                    band_arrays[name] = ds.read(idx).astype(np.float32)
                    logs.append(f"加载波段 {name} (索引 {idx})")
        else:
            if len(input_files) == len(_DEFAULT_BAND_NAMES):
                names = _DEFAULT_BAND_NAMES
                logs.append(f"多文件模式，使用默认命名: {names}")
            else:
                names = [f'band_{i+1}' for i in range(len(input_files))]
                logs.append(f"多文件模式，文件数 {len(input_files)}，使用通用命名: {names}")
            for name, fp in zip(names, input_files):
                if fp.lower().endswith(('.tif','.tiff')):
                    with rasterio.open(fp) as ds:
                        band_arrays[name] = ds.read(1).astype(np.float32)
                elif fp.lower().endswith('.npy'):
                    band_arrays[name] = np.load(fp).astype(np.float32)
                else:
                    raise ValueError(f"不支持的文件格式：{fp}")
                logs.append(f"加载波段 {name}: {fp}")

        # 计算各模块特征
        results: Dict[str, Any] = {}

        # 光谱指数
        if 'nir' in band_arrays and 'red' in band_arrays:
            results['ndvi'] = calculate_ndvi(band_arrays['nir'], band_arrays['red'])
            results['msavi'] = calculate_msavi(band_arrays['nir'], band_arrays['red'])
            logs.append("计算 NDVI, MSAVI")
        if all(b in band_arrays for b in ('nir','red','blue')):
            results['evi'] = calculate_evi(band_arrays['nir'], band_arrays['red'], band_arrays['blue'])
            logs.append("计算 EVI")
        if all(b in band_arrays for b in ('green','nir')):
            results['ndwi'] = calculate_ndwi(band_arrays['green'], band_arrays['nir'])
            logs.append("计算 NDWI")
        if all(b in band_arrays for b in ('green','swir')):
            results['mndwi'] = calculate_mndwi(band_arrays['green'], band_arrays['swir'])
            logs.append("计算 MNDWI")
        if all(b in band_arrays for b in ('swir','nir')):
            results['ndbi'] = calculate_ndbi(band_arrays['swir'], band_arrays['nir'])
            logs.append("计算 NDBI")
        if all(b in band_arrays for b in ('blue','red','nir','swir')):
            results['bsi'] = calculate_bsi(band_arrays['blue'], band_arrays['red'], band_arrays['nir'], band_arrays['swir'])
            logs.append("计算 BSI")

        # 纹理
        if 'nir' in band_arrays:
            results['glcm'] = calculate_glcm_features(band_arrays['nir'])
            results['lbp'] = calculate_lbp_features(band_arrays['nir'])
            results['gabor'] = calculate_gabor_features(band_arrays['nir'])
            logs.append("计算 GLCM, LBP, Gabor")

        # PCA
        comps, var_ratio, pca_model = perform_pca(list(band_arrays.values()), n_components=3)


        results['pca'] = comps
        results['pca_variance_ratio'] = var_ratio
        logs.append("执行 PCA")

        # 形态学 & 滤波
        if 'nir' in band_arrays:
            results['morphology'] = calculate_morphological_features(band_arrays['nir'])
            results['filter_responses'] = calculate_filter_responses(band_arrays['nir'])
            logs.append("计算形态学和滤波响应")

        # 特征选择 & 多尺度
        flat_feats: Dict[str, np.ndarray] = {}
        for k, v in results.items():
            if isinstance(v, np.ndarray) and v.ndim == 2:
                flat_feats[k] = v
            elif isinstance(v, list):
                for i, comp in enumerate(v):
                    flat_feats[f"{k}_{i}"] = comp
            elif isinstance(v, dict):
                for subk, subv in v.items():
                    flat_feats[f"{k}_{subk}"] = subv
        results['selected'] = feature_selection_by_variance(flat_feats)
        results['multi_scale'] = calculate_multi_scale_features(band_arrays.get('nir', list(band_arrays.values())[0]))
        logs.append("执行特征选择和多尺度")

        # 融合 & 上下文
        results['fused'] = feature_fusion_for_segmentation(flat_feats)
        results['segmentation_feats'] = prepare_features_for_segmentation(flat_feats)
        hier = hierarchical_feature_fusion(flat_feats)
        results['hierarchical'] = hier
        results['with_context'] = add_spatial_context(results['segmentation_feats'])
        logs.append("执行特征融合和空间上下文")

        # 保存
        for name, arr in results.items():
            if isinstance(arr, np.ndarray):
                fp = os.path.join(output_dir, f"{name}.npy")
                np.save(fp, arr)
                outputs.append(fp)
            elif isinstance(arr, list):
                for i, c in enumerate(arr):
                    fp = os.path.join(output_dir, f"{name}_{i}.npy")
                    np.save(fp, c)
                    outputs.append(fp)
            elif isinstance(arr, dict):
                subd = os.path.join(output_dir, name)
                os.makedirs(subd, exist_ok=True)
                for sk, sv in arr.items():
                    fp = os.path.join(subd, f"{sk}.npy")
                    np.save(fp, sv)
                    outputs.append(fp)
        logs.append(f"保存所有特征，共 {len(outputs)} 个文件")

        return TaskResult(status="success",
                          message="特征提取完成",
                          outputs=outputs,
                          logs=logs)

    except Exception as e:
        logs.append(f"错误: {e}")
        return TaskResult(status="failure",
                          message=str(e),
                          outputs=outputs,
                          logs=logs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="遥感影像特征提取与增强模块")
    parser.add_argument("-i","--input", nargs="+", required=True,
                        help="单文件或多文件模式：GeoTIFF/.npy")
    parser.add_argument("-o","--output", required=True,
                        help="特征输出目录")
    args = parser.parse_args()
    result = run(args.input, args.output)
    print(result)

#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
文件: engine.py
模块: src.processing.engine
功能: 提供统一的任务调度接口（run_task）及全流程入口（run），支持 YAML 或 Python 配置
作者: 孟诣楠
版本: v1.5.1
最近更新: 2025-06-18

较上一版改进:
  1. 新增 'feature_extraction' 任务支持，注册并调度 run_feature_extraction
  2. 全流程执行中，在 image_processing 与 vector_processing 之间插入 feature_extraction
"""

import logging
import os
import sys
import importlib.util
from typing import Any, Callable, Dict
# 任务结果类型
from src.processing.task_result import TaskResult

# 各模块 run 接口导入
from src.processing.file_operations.run_file_operation import run as run_file_operation
from src.processing.file_operations.run_file_saver     import run as run_file_saver
from src.processing.image_display.run_image_display   import run as run_image_display
from src.processing.image_processing.run_image_processing import run as run_image_processing
from src.processing.feature_extraction.run_feature_extraction import run as run_feature_extraction
from src.processing.vector_processing.run_vector_processing   import run as run_vector_processing
from src.processing.classification.run_classification         import run as run_classification
from src.processing.accuracy_evaluation.run_evaluation        import run as run_evaluation


class RemoteSensingEngine:
    """
    处理引擎核心，支持单任务调度（run_task）和全流程执行（run）。
    """

    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        # 注册所有任务
        self.task_registry: Dict[str, Callable[..., TaskResult]] = {
            "file_operation":    run_file_operation,
            "file_saver":        run_file_saver,
            "image_display":     run_image_display,
            "image_processing":  run_image_processing,
            "feature_extraction": run_feature_extraction,
            "vector_processing": run_vector_processing,
            "classification":    run_classification,
            "evaluation":        run_evaluation,
        }

    def run_task(self, task_name: str, **kwargs) -> TaskResult:
        """
        运行指定任务。

        参数:
            task_name: 注册任务名
            kwargs:    任务参数
        返回:
            TaskResult
        """
        if task_name not in self.task_registry:
            msg = f"未知任务: {task_name}"
            self.logger.error(msg)
            return TaskResult(status="failure", message=msg, outputs=[], logs=[msg])

        func = self.task_registry[task_name]
        self.logger.info(f"开始执行任务 [{task_name}]，参数: {kwargs}")
        try:
            # 对于 feature_extraction 任务，不传 config 关键字
            if task_name == "feature_extraction":
                result = func(kwargs.get("input_files"), kwargs.get("output_dir"))
            else:
                result = func(config=self.config, **kwargs)
            self.logger.info(f"任务 [{task_name}] 执行成功")
            return result
        except Exception as e:
            err = f"任务 [{task_name}] 执行失败: {e}"
            self.logger.exception(err)
            return TaskResult(status="failure", message=str(e), outputs=[], logs=[err])

    def run(self) -> Dict[str, TaskResult]:
        """
        一键执行全流程：按预定义顺序依次执行所有模块。

        返回:
            各任务名到 TaskResult 的映射
        """
        sequence = [
            "file_operation",
            "file_saver",
            "image_display",
            "image_processing",
            "feature_extraction",
            "vector_processing",
            "classification",
            "evaluation",
        ]
        results: Dict[str, TaskResult] = {}
        for task in sequence:
            params = getattr(self.config, f"{task}_params", {})
            res = self.run_task(task, **params)
            results[task] = res
            if res.status != "success":
                self.logger.warning(f"全流程中断于 [{task}]，状态: {res.status}")
                break
        return results


def load_config(path: str = None) -> Any:
    """
    加载配置:
      - 如果指定 YAML 文件，解析为 dict
      - 如果指定 Python 脚本 (.py)，导入并读取 Config 对象
      - 如果未指定，从根目录 config.py 导入
    """
    if path:
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.yaml', '.yml'):
            import yaml
            with open(path, 'r', encoding='utf-8') as f:
                cfg_dict = yaml.safe_load(f)
            class EngineConfig:
                def __init__(self, d): self.__dict__.update(d)
            return EngineConfig(cfg_dict)
        elif ext == '.py':
            spec = importlib.util.spec_from_file_location('config', path)
            cfg_mod = importlib.util.module_from_spec(spec)
            sys.modules['config'] = cfg_mod
            spec.loader.exec_module(cfg_mod)
            return getattr(cfg_mod, 'config', cfg_mod)
        else:
            raise ValueError(f"不支持的配置文件格式: {ext}")
    # 默认加载根目录下的 config.py
    try:
        import config as cfg_mod
        return getattr(cfg_mod, 'config', cfg_mod)
    except ImportError:
        raise FileNotFoundError("未指定配置文件，且根目录缺少 config.py")


def run():
    """
    CLI 入口：支持单任务或全流程执行。
    如果未指定 --config，将尝试导入根目录 config.py。
    """
    import argparse

    parser = argparse.ArgumentParser(description="遥感图像处理引擎 CLI")
    parser.add_argument("--config", "-c", help="配置文件路径，YAML 或 Python 脚本")
    parser.add_argument(
        "--task", "-t",
        choices=list(RemoteSensingEngine(load_config()).task_registry.keys()),
        help="指定单个任务，不指定则执行全流程"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    engine = RemoteSensingEngine(config)

    if args.task:
        params = getattr(config, f"{args.task}_params", {})
        result = engine.run_task(args.task, **params)
        print(result)
    else:
        summary = engine.run()
        for name, res in summary.items():
            print(f"{name}: {res.status} — {res.message}")


if __name__ == "__main__":
    run()

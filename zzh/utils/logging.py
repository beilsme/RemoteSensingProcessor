# -*- coding: utf-8 -*-
"""
RemoteSensingProcessor – Logging Helper
---------------------------------------
统一项目日志格式，方便快速定位问题。

Created : 2025-06-17
Author  : 张子涵
Version : 0.1.0
"""
import logging
from pathlib import Path
from datetime import datetime


LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
DATE_FMT   = "%Y-%m-%d %H:%M:%S"


def init_logger(name: str = "RSP", level: int = logging.INFO,
                log_dir: str | None = None) -> logging.Logger:
    """初始化并返回 logger；若指定 log_dir，则同时写文件"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger  # 已初始化
    logger.setLevel(level)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FMT))
    logger.addHandler(console)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_file = Path(log_dir) / f"{datetime.now():%Y%m%d}.log"
        file_hdl = logging.FileHandler(log_file, encoding="utf-8")
        file_hdl.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FMT))
        logger.addHandler(file_hdl)

    return logger

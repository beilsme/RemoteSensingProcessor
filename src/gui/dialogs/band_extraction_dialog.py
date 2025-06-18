#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
文件: band_extraction_dialog.py
模块: src.gui.dialogs.band_extraction_dialog
功能: 为波段提取任务提供参数输入对话框
"""
from PyQt6.QtWidgets import (
    QDialog, QFileDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton
)

class BandExtractionDialog(QDialog):
    """简单的波段提取参数对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("波段提取参数")

        layout = QVBoxLayout(self)

        # 输入文件
        file_layout = QHBoxLayout()
        file_layout.addWidget(QLabel("输入文件:"))
        self.in_edit = QLineEdit()
        file_layout.addWidget(self.in_edit)
        browse_in = QPushButton("浏览…")
        browse_in.clicked.connect(self._browse_input)
        file_layout.addWidget(browse_in)
        layout.addLayout(file_layout)

        # 波段
        band_layout = QHBoxLayout()
        band_layout.addWidget(QLabel("波段(逗号分隔):"))
        self.band_edit = QLineEdit("1,2,3")
        band_layout.addWidget(self.band_edit)
        layout.addLayout(band_layout)

        # 输出目录
        out_layout = QHBoxLayout()
        out_layout.addWidget(QLabel("输出目录:"))
        self.out_edit = QLineEdit()
        out_layout.addWidget(self.out_edit)
        browse_out = QPushButton("浏览…")
        browse_out.clicked.connect(self._browse_output)
        out_layout.addWidget(browse_out)
        layout.addLayout(out_layout)

        # 确认取消
        btn_layout = QHBoxLayout()
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(ok_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)

    def _browse_input(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择输入文件", "", "NumPy (*.npy);;Pickle (*.pkl);;All (*)")
        if path:
            self.in_edit.setText(path)

    def _browse_output(self):
        directory = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if directory:
            self.out_edit.setText(directory)

    def get_data(self):
        """返回 (input_paths, bands, output_dir)"""
        in_path = self.in_edit.text().strip()
        bands = [int(b) for b in self.band_edit.text().split(',') if b.strip()]
        out_dir = self.out_edit.text().strip()
        return [in_path], bands, out_dir

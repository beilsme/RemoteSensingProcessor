# src/plugins/demo_director/core/demo_director.py
import sys
import json
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtOpenGL import *
import numpy as np
from typing import Dict, List, Optional, Callable

class DemoDirector(QMainWindow):
    """演示导演主控制器"""

    def __init__(self, parent_window):
        super().__init__()
        self.parent_window = parent_window
        self.demo_engine = None
        self.current_scenario = None
        self.is_demo_running = False

        self.init_ui()
        self.load_scenarios()

    def init_ui(self):
        """初始化导演控制台界面"""
        self.setWindowTitle("🎬 智能演示导演控制台")
        self.setGeometry(100, 100, 400, 300)

        # 设置科幻风格样式
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1,
                    stop: 0 #0d1421, stop: 1 #1a2332);
                border: 2px solid #00ff88;
                border-radius: 10px;
            }
            QLabel {
                color: #00ff88;
                font-family: 'Courier New';
                font-weight: bold;
            }
            QPushButton {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #004d7a, stop: 1 #008793);
                border: 1px solid #00ff88;
                border-radius: 5px;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
            QPushButton:hover {
                background: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1,
                    stop: 0 #008793, stop: 1 #00ff88);
                box-shadow: 0 0 10px #00ff88;
            }
            QComboBox {
                background: #1a2332;
                border: 1px solid #00ff88;
                color: #00ff88;
                border-radius: 3px;
                padding: 5px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #00ff88;
                height: 8px;
                background: #1a2332;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #00ff88;
                border: 1px solid #008793;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
        """)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 标题
        title = QLabel("🎬 演示导演控制台")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; margin: 10px;")
        layout.addWidget(title)

        # 场景选择
        scenario_group = QGroupBox("演示场景选择")
        scenario_layout = QGridLayout(scenario_group)

        self.scenario_combo = QComboBox()
        scenario_layout.addWidget(QLabel("场景:"), 0, 0)
        scenario_layout.addWidget(self.scenario_combo, 0, 1)

        self.preview_btn = QPushButton("🔍 预览场景")
        self.start_demo_btn = QPushButton("🚀 开始演示")
        scenario_layout.addWidget(self.preview_btn, 1, 0)
        scenario_layout.addWidget(self.start_demo_btn, 1, 1)

        layout.addWidget(scenario_group)

        # 演示控制
        control_group = QGroupBox("演示控制")
        control_layout = QGridLayout(control_group)

        self.play_btn = QPushButton("▶️ 播放")
        self.pause_btn = QPushButton("⏸️ 暂停")
        self.stop_btn = QPushButton("⏹️ 停止")

        control_layout.addWidget(self.play_btn, 0, 0)
        control_layout.addWidget(self.pause_btn, 0, 1)
        control_layout.addWidget(self.stop_btn, 0, 2)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #00ff88;
                border-radius: 5px;
                text-align: center;
                background: #1a2332;
                color: #00ff88;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #008793, stop: 1 #00ff88);
                border-radius: 4px;
            }
        """)
        control_layout.addWidget(self.progress_bar, 1, 0, 1, 3)

        layout.addWidget(control_group)

        # 效果设置
        effects_group = QGroupBox("视觉效果设置")
        effects_layout = QGridLayout(effects_group)

        effects_layout.addWidget(QLabel("特效强度:"), 0, 0)
        self.effects_intensity = QSlider(Qt.Orientation.Horizontal)
        self.effects_intensity.setRange(1, 10)
        self.effects_intensity.setValue(7)
        effects_layout.addWidget(self.effects_intensity, 0, 1)

        effects_layout.addWidget(QLabel("解说速度:"), 1, 0)
        self.narration_speed = QSlider(Qt.Orientation.Horizontal)
        self.narration_speed.setRange(1, 10)
        self.narration_speed.setValue(5)
        effects_layout.addWidget(self.narration_speed, 1, 1)

        effects_layout.addWidget(QLabel("观众模式:"), 2, 0)
        self.audience_mode = QComboBox()
        self.audience_mode.addItems(["专业版", "科普版", "商务版"])
        effects_layout.addWidget(self.audience_mode, 2, 1)

        layout.addWidget(effects_group)

        # 连接信号
        self.connect_signals()

    def connect_signals(self):
        """连接信号槽"""
        self.start_demo_btn.clicked.connect(self.start_demo)
        self.play_btn.clicked.connect(self.play_demo)
        self.pause_btn.clicked.connect(self.pause_demo)
        self.stop_btn.clicked.connect(self.stop_demo)
        self.preview_btn.clicked.connect(self.preview_scenario)

    def load_scenarios(self):
        """加载演示场景"""
        scenarios = [
            "🏙️ 城市热岛效应分析",
            "🌱 植被覆盖度监测",
            "🌊 洪水灾害评估",
            "🔥 森林火灾监测",
            "🏔️ 冰川变化分析",
            "🌾 农作物长势监测"
        ]
        self.scenario_combo.addItems(scenarios)

    def start_demo(self):
        """开始演示"""
        if not self.is_demo_running:
            scenario_name = self.scenario_combo.currentText()

            # 创建演示引擎
            from .demo_engine import DemoEngine
            self.demo_engine = DemoEngine(
                parent_window=self.parent_window,
                effects_intensity=self.effects_intensity.value(),
                narration_speed=self.narration_speed.value(),
                audience_mode=self.audience_mode.currentText()
            )

            # 加载场景
            success = self.demo_engine.load_scenario(scenario_name)
            if success:
                self.demo_engine.progress_updated.connect(self.update_progress)
                self.demo_engine.demo_finished.connect(self.on_demo_finished)

                # 启动演示
                self.demo_engine.start_demo()
                self.is_demo_running = True
                self.update_button_states()

                QMessageBox.information(self, "演示开始", f"正在启动 {scenario_name} 演示...")
            else:
                QMessageBox.warning(self, "错误", "场景加载失败！")

    def play_demo(self):
        """播放演示"""
        if self.demo_engine:
            self.demo_engine.play()

    def pause_demo(self):
        """暂停演示"""
        if self.demo_engine:
            self.demo_engine.pause()

    def stop_demo(self):
        """停止演示"""
        if self.demo_engine:
            self.demo_engine.stop()
            self.is_demo_running = False
            self.update_button_states()
            self.progress_bar.setValue(0)

    def preview_scenario(self):
        """预览场景"""
        scenario_name = self.scenario_combo.currentText()
        preview_dialog = ScenarioPreviewDialog(scenario_name, self)
        preview_dialog.exec()

    def update_progress(self, value):
        """更新进度条"""
        self.progress_bar.setValue(value)

    def on_demo_finished(self):
        """演示完成"""
        self.is_demo_running = False
        self.update_button_states()
        QMessageBox.information(self, "演示完成", "演示已成功完成！")

    def update_button_states(self):
        """更新按钮状态"""
        self.start_demo_btn.setEnabled(not self.is_demo_running)
        self.play_btn.setEnabled(self.is_demo_running)
        self.pause_btn.setEnabled(self.is_demo_running)
        self.stop_btn.setEnabled(self.is_demo_running)


class ScenarioPreviewDialog(QDialog):
    """场景预览对话框"""

    def __init__(self, scenario_name, parent=None):
        super().__init__(parent)
        self.scenario_name = scenario_name
        self.init_ui()

    def init_ui(self):
        """初始化预览界面"""
        self.setWindowTitle(f"场景预览 - {self.scenario_name}")
        self.setGeometry(200, 200, 600, 400)

        layout = QVBoxLayout(self)

        # 场景描述
        desc_label = QLabel(self.get_scenario_description())
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #00ff88; font-size: 12px; padding: 10px;")
        layout.addWidget(desc_label)

        # 预览图片区域
        preview_area = QLabel("场景预览图将在这里显示")
        preview_area.setMinimumHeight(200)
        preview_area.setStyleSheet("""
            border: 2px dashed #00ff88;
            color: #00ff88;
            font-size: 14px;
        """)
        preview_area.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(preview_area)

        # 按钮
        button_layout = QHBoxLayout()
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        button_layout.addStretch()
        button_layout.addWidget(close_btn)

        layout.addLayout(button_layout)

    def get_scenario_description(self):
        """获取场景描述"""
        descriptions = {
            "🏙️ 城市热岛效应分析": """
            演示流程：
            1. 卫星热红外影像加载，展示城市全貌
            2. 温度数据以动态热力图形式覆盖
            3. 历史数据时间序列回放，展示热岛效应变化趋势  
            4. 粒子效果模拟热量扩散过程
            5. AI解说分析结果和环境影响
            
            预计演示时间：3-5分钟
            """,
            "🌱 植被覆盖度监测": """
            演示流程：
            1. 多光谱影像加载，春夏秋冬四季快速切换
            2. NDVI计算过程可视化，公式动画展示
            3. 植被指数以绿色波浪形式动态展现
            4. 时间序列图表实时生成
            5. 结果以3D柱状图显示各区域植被健康度
            
            预计演示时间：4-6分钟
            """,
            "🌊 洪水灾害评估": """
            演示流程：
            1. 正常水体以平静蓝色展示
            2. 暴雨动画效果，雨滴撞击地面
            3. 水体范围动态扩张，颜色渐变警示
            4. 受灾区域红色闪烁警报
            5. 损失评估数据滚动显示
            
            预计演示时间：3-4分钟
            """
        }

        return descriptions.get(self.scenario_name, "暂无描述")
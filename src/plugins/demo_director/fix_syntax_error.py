# fix_syntax_error.py
import os

def fix_visual_effects_syntax():
    """修复visual_effects.py的语法错误"""

    file_path = 'effects/visual_effects.py'

    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return

    # 重新创建一个干净的visual_effects.py
    clean_code = '''from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *
import numpy as np
import time
from typing import Dict, Any

class VisualEffectsManager(QObject):
    """视觉效果管理器"""
    
    def __init__(self, parent_window, intensity=7):
        super().__init__()
        self.parent_window = parent_window
        self.intensity = intensity
        self.active_effects = {}
        
    def trigger_effect(self, effect_name: str, params: dict):
        """触发视觉效果"""
        print(f"✨ 触发视觉效果: {effect_name}")
        effect_map = {
            "data_load": self.effect_data_load,
            "heatmap_overlay": self.effect_heatmap_overlay,
            "time_animation": self.effect_time_animation,
            "report_generation": self.effect_report_generation,
            "seasonal_change": self.effect_seasonal_change,
            "ndvi_calculation": self.effect_ndvi_calculation,
            "wave_animation": self.effect_wave_animation,
            "3d_visualization": self.effect_3d_visualization,
            "water_growth": self.effect_water_growth,
            "alert_blink": self.effect_alert_blink,
            "damage_calculation": self.effect_damage_calculation,
        }
        
        if effect_name in effect_map:
            effect_map[effect_name](params)
            
    def effect_data_load(self, params):
        """数据加载效果"""
        overlay = LoadingOverlay(self.parent_window, params)
        overlay.show()
        self.active_effects['data_load'] = overlay
        
    def effect_heatmap_overlay(self, params):
        """热力图覆盖效果"""
        heatmap_effect = HeatmapEffect(self.parent_window, params)
        heatmap_effect.start_animation()
        self.active_effects['heatmap'] = heatmap_effect
        
    def effect_time_animation(self, params):
        """时间序列动画"""
        time_effect = TimeSeriesEffect(self.parent_window, params)
        time_effect.start_animation()
        self.active_effects['time_series'] = time_effect
        
    def effect_report_generation(self, params):
        """报告生成效果"""
        report_effect = ReportGenerationEffect(self.parent_window, params)
        report_effect.start_animation()
        self.active_effects['report'] = report_effect
        
    def effect_seasonal_change(self, params):
        """季节变换效果"""
        seasonal_effect = SeasonalChangeEffect(self.parent_window, params)
        seasonal_effect.start_animation()
        self.active_effects['seasonal'] = seasonal_effect
        
    def effect_ndvi_calculation(self, params):
        """NDVI计算效果"""
        ndvi_effect = NDVICalculationEffect(self.parent_window, params)
        ndvi_effect.start_animation()
        self.active_effects['ndvi'] = ndvi_effect
        
    def effect_wave_animation(self, params):
        """波浪动画效果"""
        wave_effect = WaveAnimationEffect(self.parent_window, params)
        wave_effect.start_animation()
        self.active_effects['wave'] = wave_effect
        
    def effect_3d_visualization(self, params):
        """3D可视化效果"""
        viz3d_effect = Visualization3DEffect(self.parent_window, params)
        viz3d_effect.start_animation()
        self.active_effects['3d_viz'] = viz3d_effect
        
    def effect_water_growth(self, params):
        """水体增长效果"""
        water_effect = WaterGrowthEffect(self.parent_window, params)
        water_effect.start_animation()
        self.active_effects['water_growth'] = water_effect
        
    def effect_alert_blink(self, params):
        """警报闪烁效果"""
        alert_effect = AlertBlinkEffect(self.parent_window, params)
        alert_effect.start_animation()
        self.active_effects['alert'] = alert_effect
        
    def effect_damage_calculation(self, params):
        """损失计算效果"""
        damage_effect = DamageCalculationEffect(self.parent_window, params)
        damage_effect.start_animation()
        self.active_effects['damage'] = damage_effect
        
    def stop_all_effects(self):
        """停止所有效果"""
        for effect in self.active_effects.values():
            if hasattr(effect, 'stop'):
                effect.stop()
        self.active_effects.clear()


class BaseEffect(QWidget):
    """特效基类"""
    
    def __init__(self, parent_window, params):
        super().__init__(parent_window)
        self.parent_window = parent_window
        self.params = params
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation)
        self.frame_count = 0
        self.is_running = False
        
        # 设置为覆盖层
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        if parent_window:
            self.setGeometry(parent_window.geometry())
        
    def start_animation(self):
        """开始动画"""
        self.is_running = True
        self.frame_count = 0
        self.animation_timer.start(16)  # 60 FPS
        self.show()
        
    def stop(self):
        """停止动画"""
        self.is_running = False
        self.animation_timer.stop()
        self.hide()
        
    def update_animation(self):
        """更新动画帧"""
        if not self.is_running:
            return
            
        self.frame_count += 1
        self.update()
        
        # 子类实现具体动画逻辑
        self.animate_frame()
        
    def animate_frame(self):
        """动画帧更新 - 子类重写"""
        pass


class LoadingOverlay(BaseEffect):
    """数据加载覆盖效果"""
    
    def __init__(self, parent_window, params):
        super().__init__(parent_window, params)
        self.loading_text = "正在加载卫星影像数据..."
        self.dots_count = 0
        
    def paintEvent(self, event):
        """绘制加载效果"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 半透明背景
        painter.fillRect(self.rect(), QColor(0, 0, 0, 150))
        
        # 绘制加载圆环
        center = self.rect().center()
        radius = 50
        
        painter.setPen(QPen(QColor(0, 255, 136), 4))
        
        # 旋转的圆弧
        angle = (self.frame_count * 6) % 360
        painter.drawArc(center.x() - radius, center.y() - radius, 
                       radius * 2, radius * 2, angle * 16, 120 * 16)
        
        # 加载文本
        dots = "." * (self.dots_count % 4)
        text = self.loading_text + dots
        
        painter.setPen(QColor(0, 255, 136))
        font = QFont("Courier New", 14, QFont.Weight.Bold)
        painter.setFont(font)
        
        text_rect = QRect(center.x() - 150, center.y() + 80, 300, 30)
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, text)
        
    def animate_frame(self):
        """动画帧更新"""
        if self.frame_count % 30 == 0:  # 每半秒更新点数
            self.dots_count += 1
            
        # 3秒后自动消失
        if self.frame_count > 180:
            self.stop()


# 其他特效类的简化版本
class HeatmapEffect(BaseEffect):
    def __init__(self, parent_window, params):
        super().__init__(parent_window, params)
        self.alpha = 0
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QColor(255, 0, 0))
        painter.drawText(50, 50, "🔥 热力图效果")
        
    def animate_frame(self):
        if self.frame_count > 180:
            self.stop()


class TimeSeriesEffect(BaseEffect):
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QColor(0, 255, 136))
        painter.drawText(50, 50, "📈 时间序列动画")
        
    def animate_frame(self):
        if self.frame_count > 200:
            self.stop()


class ReportGenerationEffect(BaseEffect):
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QColor(0, 255, 136))
        painter.drawText(50, 50, "📊 正在生成分析报告...")
        
    def animate_frame(self):
        if self.frame_count > 180:
            self.stop()


class SeasonalChangeEffect(BaseEffect):
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QColor(0, 255, 100))
        painter.drawText(50, 50, "🌸🌞🍂❄️ 季节变换中...")
        
    def animate_frame(self):
        if self.frame_count > 240:
            self.stop()


class NDVICalculationEffect(BaseEffect):
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QColor(0, 255, 136))
        painter.drawText(50, 50, "🌱 NDVI计算中...")
        
    def animate_frame(self):
        if self.frame_count > 200:
            self.stop()


class WaveAnimationEffect(BaseEffect):
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QColor(0, 255, 100))
        painter.drawText(50, 50, "🌊 波浪动画效果")
        
    def animate_frame(self):
        if self.frame_count > 300:
            self.stop()


class Visualization3DEffect(BaseEffect):
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QColor(100, 150, 255))
        painter.drawText(50, 50, "📈 生成3D可视化图表...")
        
    def animate_frame(self):
        if self.frame_count > 200:
            self.stop()


class WaterGrowthEffect(BaseEffect):
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QColor(0, 150, 255))
        painter.drawText(50, 50, "🌊 水体范围扩张中...")
        
    def animate_frame(self):
        if self.frame_count > 180:
            self.stop()


class AlertBlinkEffect(BaseEffect):
    def paintEvent(self, event):
        painter = QPainter(self)
        if (self.frame_count // 15) % 2:
            painter.setPen(QColor(255, 0, 0))
            painter.drawText(50, 50, "🚨 警报：受灾区域检测到！")
        
    def animate_frame(self):
        if self.frame_count > 150:
            self.stop()


class DamageCalculationEffect(BaseEffect):
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setPen(QColor(255, 200, 0))
        painter.drawText(50, 50, "💰 正在计算经济损失...")
        
    def animate_frame(self):
        if self.frame_count > 200:
            self.stop()
'''

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(clean_code)

    print("✅ visual_effects.py 已重新创建，语法错误已修复")

if __name__ == "__main__":
    fix_visual_effects_syntax()
# src/plugins/demo_director/core/demo_engine.py
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtOpenGL import *
import time
import numpy as np
from typing import Dict, Any, Optional

class DemoEngine(QObject):
    """演示引擎核心类"""

    progress_updated = pyqtSignal(int)
    demo_finished = pyqtSignal()
    effect_triggered = pyqtSignal(str, dict)

    def __init__(self, parent_window, effects_intensity=7,
                 narration_speed=5, audience_mode="专业版"):
        super().__init__()
        self.parent_window = parent_window
        self.effects_intensity = effects_intensity
        self.narration_speed = narration_speed
        self.audience_mode = audience_mode

        self.current_scenario = None
        self.demo_timeline = []
        self.current_step = 0
        self.is_playing = False
        self.is_paused = False

        # 初始化子系统
        self.init_subsystems()

    def init_subsystems(self):
        """初始化子系统"""
        from ..effects.visual_effects import VisualEffectsManager
        from ..effects.particle_system import ParticleSystem
        from ..narration.ai_narrator import AISemanticNarrator

        self.effects_manager = VisualEffectsManager(
            self.parent_window,
            intensity=self.effects_intensity
        )

        self.particle_system = ParticleSystem(self.parent_window)

        self.narrator = AISemanticNarrator(
            speed=self.narration_speed,
            mode=self.audience_mode
        )

    def load_scenario(self, scenario_name: str) -> bool:
        """加载演示场景"""
        try:
            self.current_scenario = scenario_name
            self.demo_timeline = self.create_timeline(scenario_name)
            self.current_step = 0
            return True
        except Exception as e:
            print(f"场景加载失败: {e}")
            return False

    def create_timeline(self, scenario_name: str) -> list:
        """创建演示时间轴"""
        timelines = {
            "🏙️ 城市热岛效应分析": self.create_urban_heat_timeline(),
            "🌱 植被覆盖度监测": self.create_vegetation_timeline(),
            "🌊 洪水灾害评估": self.create_flood_timeline(),
        }

        return timelines.get(scenario_name, [])

    def create_urban_heat_timeline(self) -> list:
        """创建城市热岛效应演示时间轴"""
        return [
            {
                "duration": 2.0,
                "action": "load_data",
                "params": {"data_type": "thermal", "effect": "zoom_in"},
                "narration": "现在让我们来分析城市热岛效应。首先加载热红外卫星影像数据。"
            },
            {
                "duration": 3.0,
                "action": "apply_heatmap",
                "params": {"animation": "gradient_overlay", "colors": "thermal"},
                "narration": "通过热力图可视化，我们可以清晰地看到城市不同区域的温度分布。"
            },
            {
                "duration": 4.0,
                "action": "time_series_animation",
                "params": {"type": "historical_data", "speed": "medium"},
                "narration": "这是过去十年的温度变化趋势，注意观察热岛效应的演变过程。"
            },
            {
                "duration": 3.0,
                "action": "particle_simulation",
                "params": {"type": "heat_diffusion", "intensity": "high"},
                "narration": "粒子效果模拟显示了热量在城市中的扩散规律。"
            },
            {
                "duration": 2.0,
                "action": "generate_report",
                "params": {"type": "statistics", "animation": "count_up"},
                "narration": "分析完成。城市中心比郊区平均高温3.8度，建议增加绿化面积。"
            }
        ]

    def create_vegetation_timeline(self) -> list:
        """创建植被监测演示时间轴"""
        return [
            {
                "duration": 2.0,
                "action": "load_data",
                "params": {"data_type": "multispectral", "effect": "fade_in"},
                "narration": "加载多光谱卫星影像，准备进行植被覆盖度分析。"
            },
            {
                "duration": 3.0,
                "action": "seasonal_transition",
                "params": {"seasons": ["spring", "summer", "autumn", "winter"]},
                "narration": "观察四季植被变化，春绿夏盛秋黄冬枯的自然规律清晰可见。"
            },
            {
                "duration": 4.0,
                "action": "ndvi_calculation",
                "params": {"show_formula": True, "animation": "step_by_step"},
                "narration": "计算归一化植被指数NDVI，公式为(NIR-RED)/(NIR+RED)。"
            },
            {
                "duration": 3.0,
                "action": "wave_visualization",
                "params": {"type": "vegetation_waves", "color": "green"},
                "narration": "植被指数以绿色波浪形式展现，波峰代表植被茂盛区域。"
            },
            {
                "duration": 3.0,
                "action": "3d_chart",
                "params": {"type": "bar_chart", "data": "vegetation_health"},
                "narration": "三维柱状图显示各区域植被健康度，为生态保护提供数据支撑。"
            }
        ]

    def create_flood_timeline(self) -> list:
        """创建洪水评估演示时间轴"""
        return [
            {
                "duration": 2.0,
                "action": "load_data",
                "params": {"data_type": "water_body", "effect": "calm_blue"},
                "narration": "首先显示正常状态下的水体分布，一切都很平静。"
            },
            {
                "duration": 3.0,
                "action": "rain_animation",
                "params": {"intensity": "heavy", "sound": True},
                "narration": "突然暴雨来袭！观察雨滴撞击地面的动态效果。"
            },
            {
                "duration": 4.0,
                "action": "water_expansion",
                "params": {"animation": "dynamic_growth", "color_change": True},
                "narration": "水体范围快速扩张，颜色从蓝色变为警示红色。"
            },
            {
                "duration": 2.0,
                "action": "alert_effects",
                "params": {"type": "blinking_red", "areas": "affected_regions"},
                "narration": "受灾区域红色闪烁报警，紧急疏散通道已标出。"
            },
            {
                "duration": 3.0,
                "action": "damage_assessment",
                "params": {"animation": "rolling_numbers", "type": "economic_loss"},
                "narration": "损失评估完成：受灾面积128平方公里，经济损失约2.3亿元。"
            }
        ]

    def start_demo(self):
        """开始演示"""
        if not self.demo_timeline:
            return False

        self.is_playing = True
        self.is_paused = False
        self.current_step = 0

        # 启动演示主循环
        self.demo_timer = QTimer()
        self.demo_timer.timeout.connect(self.execute_next_step)
        self.demo_timer.start(100)  # 每100ms检查一次

        # 开始第一步
        self.execute_current_step()

        return True

    def execute_current_step(self):
        """执行当前步骤"""
        if self.current_step >= len(self.demo_timeline):
            self.finish_demo()
            return

        step = self.demo_timeline[self.current_step]

        # 执行动作
        self.execute_action(step["action"], step["params"])

        # 播放解说
        if "narration" in step:
            self.narrator.speak(step["narration"])

        # 设置步骤持续时间
        self.step_start_time = time.time()
        self.step_duration = step["duration"]

    def execute_next_step(self):
        """检查是否进入下一步"""
        if not self.is_playing or self.is_paused:
            return

        elapsed = time.time() - self.step_start_time

        # 更新进度
        if hasattr(self, 'step_duration'):
            step_progress = min(100, (elapsed / self.step_duration) * 100)
            total_progress = ((self.current_step + step_progress/100) /
                              len(self.demo_timeline)) * 100
            self.progress_updated.emit(int(total_progress))

        # 检查是否完成当前步骤
        if elapsed >= self.step_duration:
            self.current_step += 1
            if self.current_step < len(self.demo_timeline):
                self.execute_current_step()
            else:
                self.finish_demo()

    def execute_action(self, action: str, params: dict):
        """执行演示动作"""
        action_map = {
            "load_data": self.action_load_data,
            "apply_heatmap": self.action_apply_heatmap,
            "time_series_animation": self.action_time_series,
            "particle_simulation": self.action_particle_simulation,
            "generate_report": self.action_generate_report,
            "seasonal_transition": self.action_seasonal_transition,
            "ndvi_calculation": self.action_ndvi_calculation,
            "wave_visualization": self.action_wave_visualization,
            "3d_chart": self.action_3d_chart,
            "rain_animation": self.action_rain_animation,
            "water_expansion": self.action_water_expansion,
            "alert_effects": self.action_alert_effects,
            "damage_assessment": self.action_damage_assessment,
        }

        if action in action_map:
            action_map[action](params)
        else:
            print(f"未知动作: {action}")

    def action_load_data(self, params):
        """加载数据动作"""
        self.effects_manager.trigger_effect("data_load", params)

    def action_apply_heatmap(self, params):
        """应用热力图动作"""
        self.effects_manager.trigger_effect("heatmap_overlay", params)

    def action_time_series(self, params):
        """时间序列动画"""
        self.effects_manager.trigger_effect("time_animation", params)

    def action_particle_simulation(self, params):
        """粒子模拟"""
        self.particle_system.start_simulation(params["type"], params)

    def action_generate_report(self, params):
        """生成报告"""
        self.effects_manager.trigger_effect("report_generation", params)

    def action_seasonal_transition(self, params):
        """季节转换"""
        self.effects_manager.trigger_effect("seasonal_change", params)

    def action_ndvi_calculation(self, params):
        """NDVI计算"""
        self.effects_manager.trigger_effect("ndvi_calculation", params)

    def action_wave_visualization(self, params):
        """波浪可视化"""
        self.effects_manager.trigger_effect("wave_animation", params)

    def action_3d_chart(self, params):
        """3D图表"""
        self.effects_manager.trigger_effect("3d_visualization", params)

    def action_rain_animation(self, params):
        """雨滴动画"""
        self.particle_system.start_simulation("rain", params)

    def action_water_expansion(self, params):
        """水体扩张"""
        self.effects_manager.trigger_effect("water_growth", params)

    def action_alert_effects(self, params):
        """警报效果"""
        self.effects_manager.trigger_effect("alert_blink", params)

    def action_damage_assessment(self, params):
        """损失评估"""
        self.effects_manager.trigger_effect("damage_calculation", params)

    def play(self):
        """播放"""
        self.is_paused = False

    def pause(self):
        """暂停"""
        self.is_paused = True

    def stop(self):
        """停止"""
        self.is_playing = False
        self.is_paused = False
        if hasattr(self, 'demo_timer'):
            self.demo_timer.stop()
        self.effects_manager.stop_all_effects()
        self.particle_system.stop_all_simulations()
        self.narrator.stop()

    def finish_demo(self):
        """完成演示"""
        self.stop()
        self.progress_updated.emit(100)
        self.demo_finished.emit()
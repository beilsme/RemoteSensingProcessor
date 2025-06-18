# src/plugins/demo_director/effects/particle_system.py
import random
import math
from PyQt6.QtCore import *
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *

class Particle:
    """粒子类"""

    def __init__(self, x, y, vx, vy, color, size, life):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.color = color
        self.size = size
        self.life = life
        self.max_life = life

    def update(self):
        """更新粒子状态"""
        self.x += self.vx
        self.y += self.vy
        self.life -= 1

        # 重力效果
        self.vy += 0.1

    def is_alive(self):
        """检查粒子是否还活着"""
        return self.life > 0

    def get_alpha(self):
        """获取透明度"""
        return int(255 * (self.life / self.max_life))


class ParticleSystem(QWidget):
    """粒子系统"""

    def __init__(self, parent_window):
        super().__init__(parent_window)
        self.parent_window = parent_window
        self.particles = []
        self.emitters = []
        self.simulation_type = None

        # 设置为覆盖层
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setGeometry(parent_window.geometry())

        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_particles)

    def start_simulation(self, sim_type, params):
        """开始粒子模拟"""
        self.simulation_type = sim_type
        self.particles.clear()
        self.emitters.clear()

        if sim_type == "heat_diffusion":
            self.setup_heat_diffusion(params)
        elif sim_type == "rain":
            self.setup_rain_simulation(params)
        elif sim_type == "explosion":
            self.setup_explosion_effect(params)

        self.animation_timer.start(16)  # 60 FPS
        self.show()

    def setup_heat_diffusion(self, params):
        """设置热扩散效果"""
        center_x = self.width() // 2
        center_y = self.height() // 2

        # 创建热粒子发射器
        for i in range(50):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed

            # 热粒子颜色（红到黄渐变）
            heat_level = random.uniform(0, 1)
            if heat_level > 0.7:
                color = QColor(255, 100, 0)  # 橙红
            elif heat_level > 0.4:
                color = QColor(255, 200, 0)  # 黄
            else:
                color = QColor(255, 255, 100)  # 淡黄

            particle = Particle(
                x=center_x + random.uniform(-20, 20),
                y=center_y + random.uniform(-20, 20),
                vx=vx, vy=vy,
                color=color,
                size=random.uniform(3, 8),
                life=random.randint(60, 120)
            )
            self.particles.append(particle)

    def setup_rain_simulation(self, params):
        """设置雨滴模拟"""
        intensity = params.get("intensity", "medium")

        if intensity == "heavy":
            particle_count = 100
        elif intensity == "light":
            particle_count = 30
        else:
            particle_count = 60

        for i in range(particle_count):
            x = random.uniform(0, self.width())
            y = random.uniform(-100, 0)

            particle = Particle(
                x=x, y=y,
                vx=random.uniform(-1, 1),
                vy=random.uniform(5, 10),
                color=QColor(100, 150, 255),
                size=random.uniform(1, 3),
                life=random.randint(60, 180)
            )
            self.particles.append(particle)

    def setup_explosion_effect(self, params):
        """设置爆炸效果"""
        center_x = params.get("x", self.width() // 2)
        center_y = params.get("y", self.height() // 2)

        for i in range(80):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 8)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed

            colors = [
                QColor(255, 100, 100),  # 红
                QColor(255, 200, 100),  # 橙
                QColor(255, 255, 100),  # 黄
                QColor(100, 255, 100),  # 绿
            ]

            particle = Particle(
                x=center_x, y=center_y,
                vx=vx, vy=vy,
                color=random.choice(colors),
                size=random.uniform(2, 6),
                life=random.randint(30, 90)
            )
            self.particles.append(particle)

    def update_particles(self):
        """更新所有粒子"""
        # 移除死亡粒子
        self.particles = [p for p in self.particles if p.is_alive()]

        # 更新活着的粒子
        for particle in self.particles:
            particle.update()

        # 持续生成新粒子（针对某些类型）
        if self.simulation_type == "rain" and len(self.particles) < 50:
            for i in range(3):
                x = random.uniform(0, self.width())
                y = random.uniform(-50, -10)

                particle = Particle(
                    x=x, y=y,
                    vx=random.uniform(-1, 1),
                    vy=random.uniform(5, 10),
                    color=QColor(100, 150, 255),
                    size=random.uniform(1, 3),
                    life=random.randint(60, 180)
                )
                self.particles.append(particle)

        self.update()

        # 检查是否停止模拟
        if len(self.particles) == 0 and self.simulation_type != "rain":
            self.stop_all_simulations()

    def paintEvent(self, event):
        """绘制粒子"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        for particle in self.particles:
            color = QColor(particle.color)
            color.setAlpha(particle.get_alpha())

            painter.setBrush(QBrush(color))
            painter.setPen(Qt.PenStyle.NoPen)

            painter.drawEllipse(
                int(particle.x - particle.size/2),
                int(particle.y - particle.size/2),
                int(particle.size),
                int(particle.size)
            )

    def stop_all_simulations(self):
        """停止所有模拟"""
        self.animation_timer.stop()
        self.particles.clear()
        self.hide()


# src/plugins/demo_director/narration/ai_narrator.py
import threading
import time
from PyQt6.QtCore import *
from PyQt6.QtMultimedia import *

class AISemanticNarrator(QObject):
    """AI语义解说器"""

    speech_started = pyqtSignal()
    speech_finished = pyqtSignal()

    def __init__(self, speed=5, mode="专业版"):
        super().__init__()
        self.speed = speed
        self.mode = mode
        self.is_speaking = False

        # 初始化语音合成
        self.init_speech_engine()

    def init_speech_engine(self):
        """初始化语音合成引擎"""
        # 这里可以集成各种TTS引擎
        # 比如 pyttsx3, Azure Speech Services, 百度语音等
        try:
            import pyttsx3
            self.engine = pyttsx3.init()

            # 设置语音参数
            voices = self.engine.getProperty('voices')
            if voices:
                # 尝试选择中文语音
                for voice in voices:
                    if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower():
                        self.engine.setProperty('voice', voice.id)
                        break

            # 设置语速
            rate = self.engine.getProperty('rate')
            self.engine.setProperty('rate', rate + (self.speed - 5) * 20)

        except ImportError:
            print("语音合成引擎未安装，使用文本显示模式")
            self.engine = None

    def speak(self, text):
        """播放解说"""
        if self.is_speaking:
            return

        # 根据观众模式调整解说内容
        adjusted_text = self.adjust_text_for_audience(text)

        self.speech_started.emit()
        self.is_speaking = True

        if self.engine:
            # 异步播放语音
            thread = threading.Thread(target=self._speak_async, args=(adjusted_text,))
            thread.start()
        else:
            # 文本显示模式
            self._show_text_subtitle(adjusted_text)

    def _speak_async(self, text):
        """异步语音播放"""
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except:
            print(f"语音播放失败: {text}")
        finally:
            self.is_speaking = False
            self.speech_finished.emit()

    def _show_text_subtitle(self, text):
        """显示文本字幕"""
        # 创建字幕显示窗口
        subtitle_window = SubtitleWindow(text)
        subtitle_window.show()

        # 模拟语音播放时间
        duration = len(text) * 100  # 每个字符100ms
        QTimer.singleShot(duration, lambda: self._finish_subtitle(subtitle_window))

    def _finish_subtitle(self, subtitle_window):
        """完成字幕显示"""
        subtitle_window.close()
        self.is_speaking = False
        self.speech_finished.emit()

    def adjust_text_for_audience(self, text):
        """根据观众模式调整解说内容"""
        if self.mode == "科普版":
            # 添加更多解释
            technical_terms = {
                "NDVI": "归一化植被指数NDVI",
                "热红外": "热红外遥感技术",
                "像素": "图像的最小单元像素",
                "波段": "不同波长的电磁波波段"
            }

            for term, explanation in technical_terms.items():
                text = text.replace(term, explanation)

        elif self.mode == "商务版":
            # 突出商业价值
            business_phrases = {
                "分析": "深度分析",
                "监测": "智能监测",
                "效果": "显著效果",
                "结果": "精准结果"
            }

            for phrase, business_phrase in business_phrases.items():
                text = text.replace(phrase, business_phrase)

        return text

    def stop(self):
        """停止解说"""
        if self.engine and self.is_speaking:
            self.engine.stop()
        self.is_speaking = False


class SubtitleWindow(QWidget):
    """字幕显示窗口"""

    def __init__(self, text):
        super().__init__()
        self.text = text
        self.init_ui()

    def init_ui(self):
        """初始化界面"""
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # 获取屏幕尺寸
        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(0, screen.height() - 150, screen.width(), 100)

        layout = QVBoxLayout(self)

        label = QLabel(self.text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setWordWrap(True)
        label.setStyleSheet("""
            QLabel {
                background-color: rgba(0, 0, 0, 180);
                color: #00ff88;
                font-size: 16px;
                font-weight: bold;
                padding: 15px;
                border-radius: 10px;
                border: 2px solid #00ff88;
            }
        """)

        layout.addWidget(label)


# src/plugins/demo_director/plugin_manager.py
class DemoPluginManager:
    """演示插件管理器"""

    def __init__(self, main_window):
        self.main_window = main_window
        self.demo_director = None

    def install_plugin(self):
        """安装演示插件"""
        # 在主窗口工具栏添加演示按钮
        self.add_demo_button()

        # 在菜单栏添加演示菜单
        self.add_demo_menu()

        print("智能演示导演插件已成功安装！")

    def add_demo_button(self):
        """添加演示按钮到工具栏"""
        demo_action = QAction("🎬 演示模式", self.main_window)
        demo_action.setStatusTip("启动智能演示导演")
        demo_action.setShortcut("Ctrl+D")
        demo_action.triggered.connect(self.launch_demo_director)

        # 获取主工具栏并添加按钮
        toolbar = self.main_window.findChild(QToolBar)
        if toolbar:
            toolbar.addSeparator()
            toolbar.addAction(demo_action)
        else:
            # 如果没有工具栏，创建一个
            toolbar = self.main_window.addToolBar("演示工具栏")
            toolbar.addAction(demo_action)

    def add_demo_menu(self):
        """添加演示菜单"""
        menubar = self.main_window.menuBar()

        demo_menu = menubar.addMenu("演示(&D)")

        # 启动演示导演
        launch_action = QAction("🎬 启动演示导演", self.main_window)
        launch_action.setShortcut("Ctrl+D")
        launch_action.triggered.connect(self.launch_demo_director)
        demo_menu.addAction(launch_action)

        demo_menu.addSeparator()

        # 快速演示场景
        scenarios = [
            ("🏙️ 城市热岛分析", "urban_heat"),
            ("🌱 植被监测", "vegetation"),
            ("🌊 洪水评估", "flood"),
            ("🔥 火灾监测", "fire"),
            ("🏔️ 冰川变化", "glacier"),
            ("🌾 农作物监测", "agriculture")
        ]

        quick_demo_menu = demo_menu.addMenu("快速演示")
        for name, scenario_id in scenarios:
            action = QAction(name, self.main_window)
            action.triggered.connect(lambda checked, sid=scenario_id: self.quick_demo(sid))
            quick_demo_menu.addAction(action)

        demo_menu.addSeparator()

        # 演示设置
        settings_action = QAction("⚙️ 演示设置", self.main_window)
        settings_action.triggered.connect(self.show_demo_settings)
        demo_menu.addAction(settings_action)

        # 帮助
        help_action = QAction("❓ 演示帮助", self.main_window)
        help_action.triggered.connect(self.show_demo_help)
        demo_menu.addAction(help_action)

    def launch_demo_director(self):
        """启动演示导演"""
        from .core.demo_director import DemoDirector

        if not self.demo_director:
            self.demo_director = DemoDirector(self.main_window)

        self.demo_director.show()
        self.demo_director.raise_()
        self.demo_director.activateWindow()

    def quick_demo(self, scenario_id):
        """快速演示"""
        from .core.demo_engine import DemoEngine

        # 场景映射
        scenario_map = {
            "urban_heat": "🏙️ 城市热岛效应分析",
            "vegetation": "🌱 植被覆盖度监测",
            "flood": "🌊 洪水灾害评估",
            "fire": "🔥 森林火灾监测",
            "glacier": "🏔️ 冰川变化分析",
            "agriculture": "🌾 农作物长势监测"
        }

        scenario_name = scenario_map.get(scenario_id, "🏙️ 城市热岛效应分析")

        # 直接启动演示
        demo_engine = DemoEngine(self.main_window)
        success = demo_engine.load_scenario(scenario_name)

        if success:
            demo_engine.start_demo()
            QMessageBox.information(
                self.main_window,
                "快速演示",
                f"正在启动 {scenario_name} 快速演示..."
            )
        else:
            QMessageBox.warning(
                self.main_window,
                "演示失败",
                "场景加载失败，请检查数据文件！"
            )

    def show_demo_settings(self):
        """显示演示设置"""
        settings_dialog = DemoSettingsDialog(self.main_window)
        settings_dialog.exec()

    def show_demo_help(self):
        """显示演示帮助"""
        help_dialog = DemoHelpDialog(self.main_window)
        help_dialog.exec()


class DemoSettingsDialog(QDialog):
    """演示设置对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("演示设置")
        self.setFixedSize(400, 300)

        layout = QVBoxLayout(self)

        # 视觉效果设置
        effects_group = QGroupBox("视觉效果设置")
        effects_layout = QFormLayout(effects_group)

        self.effects_quality = QComboBox()
        self.effects_quality.addItems(["高质量", "中等质量", "流畅优先"])
        effects_layout.addRow("效果质量:", self.effects_quality)

        self.particle_count = QSlider(Qt.Orientation.Horizontal)
        self.particle_count.setRange(50, 500)
        self.particle_count.setValue(200)
        effects_layout.addRow("粒子数量:", self.particle_count)

        self.animation_speed = QSlider(Qt.Orientation.Horizontal)
        self.animation_speed.setRange(1, 10)
        self.animation_speed.setValue(5)
        effects_layout.addRow("动画速度:", self.animation_speed)

        layout.addWidget(effects_group)

        # 音频设置
        audio_group = QGroupBox("音频设置")
        audio_layout = QFormLayout(audio_group)

        self.enable_narration = QCheckBox("启用AI解说")
        self.enable_narration.setChecked(True)
        audio_layout.addRow(self.enable_narration)

        self.voice_speed = QSlider(Qt.Orientation.Horizontal)
        self.voice_speed.setRange(1, 10)
        self.voice_speed.setValue(5)
        audio_layout.addRow("解说语速:", self.voice_speed)

        self.enable_sound_effects = QCheckBox("启用音效")
        self.enable_sound_effects.setChecked(True)
        audio_layout.addRow(self.enable_sound_effects)

        layout.addWidget(audio_group)

        # 按钮
        button_layout = QHBoxLayout()

        save_btn = QPushButton("保存设置")
        save_btn.clicked.connect(self.save_settings)

        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.close)

        reset_btn = QPushButton("恢复默认")
        reset_btn.clicked.connect(self.reset_settings)

        button_layout.addWidget(save_btn)
        button_layout.addWidget(cancel_btn)
        button_layout.addWidget(reset_btn)

        layout.addLayout(button_layout)

    def save_settings(self):
        """保存设置"""
        settings = {
            'effects_quality': self.effects_quality.currentText(),
            'particle_count': self.particle_count.value(),
            'animation_speed': self.animation_speed.value(),
            'enable_narration': self.enable_narration.isChecked(),
            'voice_speed': self.voice_speed.value(),
            'enable_sound_effects': self.enable_sound_effects.isChecked()
        }

        # 保存到配置文件
        import json
        try:
            with open('demo_settings.json', 'w', encoding='utf-8') as f:
                json.dump(settings, f, ensure_ascii=False, indent=2)
            QMessageBox.information(self, "设置", "设置已保存！")
            self.close()
        except Exception as e:
            QMessageBox.warning(self, "错误", f"设置保存失败: {e}")

    def reset_settings(self):
        """重置为默认设置"""
        self.effects_quality.setCurrentIndex(0)
        self.particle_count.setValue(200)
        self.animation_speed.setValue(5)
        self.enable_narration.setChecked(True)
        self.voice_speed.setValue(5)
        self.enable_sound_effects.setChecked(True)


class DemoHelpDialog(QDialog):
    """演示帮助对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        """初始化界面"""
        self.setWindowTitle("演示插件帮助")
        self.setGeometry(200, 200, 600, 500)

        layout = QVBoxLayout(self)

        # 创建标签页
        tab_widget = QTabWidget()

        # 基本使用
        basic_tab = QWidget()
        basic_layout = QVBoxLayout(basic_tab)

        basic_text = QTextEdit()
        basic_text.setReadOnly(True)
        basic_text.setHtml("""
        <h2>🎬 智能演示导演插件使用指南</h2>
        
        <h3>基本功能</h3>
        <ul>
        <li><b>一键演示</b>：选择场景后自动生成完整演示流程</li>
        <li><b>电影级特效</b>：粒子系统、动态渲染、过渡动画</li>
        <li><b>AI智能解说</b>：根据观众类型自动调整解说内容</li>
        <li><b>实时控制</b>：播放、暂停、停止、调速等功能</li>
        </ul>
        
        <h3>快速开始</h3>
        <ol>
        <li>点击工具栏上的 🎬 演示模式 按钮</li>
        <li>在演示导演控制台中选择场景</li>
        <li>点击"开始演示"启动自动演示</li>
        <li>使用控制按钮管理演示进度</li>
        </ol>
        
        <h3>快捷键</h3>
        <ul>
        <li><b>Ctrl+D</b>：启动演示导演</li>
        <li><b>空格键</b>：播放/暂停演示</li>
        <li><b>Esc</b>：停止演示</li>
        </ul>
        """)
        basic_layout.addWidget(basic_text)
        tab_widget.addTab(basic_tab, "基本使用")

        # 高级功能
        advanced_tab = QWidget()
        advanced_layout = QVBoxLayout(advanced_tab)

        advanced_text = QTextEdit()
        advanced_text.setReadOnly(True)
        advanced_text.setHtml("""
        <h2>🚀 高级功能</h2>
        
        <h3>自定义演示场景</h3>
        <p>您可以创建自己的演示场景：</p>
        <ol>
        <li>编辑场景配置文件</li>
        <li>定义时间轴和动作序列</li>
        <li>添加自定义解说文本</li>
        <li>配置视觉效果参数</li>
        </ol>
        
        <h3>观众模式</h3>
        <ul>
        <li><b>专业版</b>：包含技术细节，适合专业人士</li>
        <li><b>科普版</b>：增加解释说明，适合普通观众</li>
        <li><b>商务版</b>：突出商业价值，适合客户演示</li>
        </ul>
        
        <h3>效果调节</h3>
        <ul>
        <li><b>特效强度</b>：控制视觉效果的复杂度</li>
        <li><b>解说速度</b>：调整AI解说的语速</li>
        <li><b>粒子数量</b>：影响性能和视觉效果</li>
        </ul>
        """)
        advanced_layout.addWidget(advanced_text)
        tab_widget.addTab(advanced_tab, "高级功能")

        # 故障排除
        troubleshoot_tab = QWidget()
        troubleshoot_layout = QVBoxLayout(troubleshoot_tab)

        troubleshoot_text = QTextEdit()
        troubleshoot_text.setReadOnly(True)
        troubleshoot_text.setHtml("""
        <h2>🔧 故障排除</h2>
        
        <h3>常见问题</h3>
        
        <h4>Q: 演示过程中出现卡顿？</h4>
        <p>A: 尝试以下解决方案：</p>
        <ul>
        <li>降低特效强度设置</li>
        <li>减少粒子数量</li>
        <li>关闭其他耗资源的程序</li>
        <li>选择"流畅优先"模式</li>
        </ul>
        
        <h4>Q: 没有声音或解说？</h4>
        <p>A: 检查以下设置：</p>
        <ul>
        <li>确认系统音量已开启</li>
        <li>检查"启用AI解说"选项</li>
        <li>安装语音合成组件</li>
        <li>尝试重启演示插件</li>
        </ul>
        
        <h4>Q: 演示场景加载失败？</h4>
        <p>A: 可能的原因：</p>
        <ul>
        <li>缺少示例数据文件</li>
        <li>文件路径不正确</li>
        <li>权限问题</li>
        <li>依赖库未安装</li>
        </ul>
        
        <h3>技术支持</h3>
        <p>如果问题仍然存在，请联系技术支持团队。</p>
        """)
        troubleshoot_layout.addWidget(troubleshoot_text)
        tab_widget.addTab(troubleshoot_tab, "故障排除")

        layout.addWidget(tab_widget)

        # 关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

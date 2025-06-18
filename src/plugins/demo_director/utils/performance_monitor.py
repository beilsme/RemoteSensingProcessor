# src/plugins/demo_director/utils/performance_monitor.py
import time
import psutil
from PyQt6.QtCore import *

class PerformanceMonitor(QObject):
    """性能监控器"""

    performance_update = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.monitoring = False
        self.start_time = None
        self.frame_times = []

        self.monitor_timer = QTimer()
        self.monitor_timer.timeout.connect(self.update_performance)

    def start_monitoring(self):
        """开始性能监控"""
        self.monitoring = True
        self.start_time = time.time()
        self.frame_times.clear()
        self.monitor_timer.start(1000)  # 每秒更新

    def stop_monitoring(self):
        """停止性能监控"""
        self.monitoring = False
        self.monitor_timer.stop()

    def record_frame_time(self, frame_time: float):
        """记录帧时间"""
        self.frame_times.append(frame_time)

        # 只保留最近100帧的数据
        if len(self.frame_times) > 100:
            self.frame_times.pop(0)

    def update_performance(self):
        """更新性能数据"""
        if not self.monitoring:
            return

        current_time = time.time()

        # 计算FPS
        fps = 0
        if self.frame_times:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            if avg_frame_time > 0:
                fps = 1.0 / avg_frame_time

        # 获取系统性能数据
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()

        performance_data = {
            'fps': fps,
            'cpu_percent': cpu_percent,
            'memory_percent': memory_info.percent,
            'memory_used_mb': memory_info.used / 1024 / 1024,
            'runtime': current_time - self.start_time if self.start_time else 0
        }

        self.performance_update.emit(performance_data)


# 主窗口集成示例
def integrate_demo_plugin(main_window):
    """在主窗口中集成演示插件"""

    # 创建插件管理器
    plugin_manager = DemoPluginManager(main_window)

    # 安装插件
    plugin_manager.install_plugin()

    # 添加到主窗口
    main_window.demo_plugin_manager = plugin_manager

    print("🎬 智能演示导演插件安装完成！")
    print("使用 Ctrl+D 快捷键或点击工具栏按钮启动演示。")


# 使用示例
if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication, QMainWindow

    app = QApplication(sys.argv)

    # 创建主窗口（这里用空窗口模拟）
    main_window = QMainWindow()
    main_window.setWindowTitle("遥感图像处理系统")
    main_window.setGeometry(100, 100, 1200, 800)

    # 集成演示插件
    integrate_demo_plugin(main_window)

    main_window.show()

    sys.exit(app.exec_())
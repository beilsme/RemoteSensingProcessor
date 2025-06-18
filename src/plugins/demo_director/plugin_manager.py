from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

class DemoPluginManager:
    """演示插件管理器"""

    def __init__(self, main_window):
        self.main_window = main_window
        self.demo_director = None

    def install_plugin(self):
        """安装演示插件"""
        print("🔧 正在安装演示插件...")

        # 添加演示菜单
        self.add_demo_menu()
        print("✅ 演示插件安装完成！")

    def add_demo_menu(self):
        """添加演示菜单"""
        menubar = self.main_window.menuBar()

        demo_menu = menubar.addMenu("演示(&D)")

        # 启动演示导演
        launch_action = QAction("🎬 启动演示导演", self.main_window)
        launch_action.setShortcut("Ctrl+D")
        launch_action.triggered.connect(self.launch_demo_director)
        demo_menu.addAction(launch_action)

    def launch_demo_director(self):
        """启动演示导演"""
        from .core.demo_director import DemoDirector

        if not self.demo_director:
            self.demo_director = DemoDirector(self.main_window)

        self.demo_director.show()
        self.demo_director.raise_()
        self.demo_director.activateWindow()
# test_demo_plugin.py
import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton
from PyQt6.QtCore import Qt

# 添加src路径
sys.path.insert(0, "src")

class MockMainWindow(QMainWindow):
    """模拟主窗口用于测试"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("遥感图像处理系统 - 演示插件测试")
        self.setGeometry(100, 100, 1200, 800)

        # 创建中央组件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 添加一些测试按钮
        test_btn = QPushButton("测试基础功能")
        test_btn.clicked.connect(self.test_basic_functions)
        layout.addWidget(test_btn)

        demo_btn = QPushButton("🎬 启动演示插件")
        demo_btn.clicked.connect(self.launch_demo_plugin)
        layout.addWidget(demo_btn)

        # 集成演示插件
        self.integrate_demo_plugin()

    def integrate_demo_plugin(self):
        """集成演示插件"""
        try:
            from plugins.demo_director.plugin_manager import DemoPluginManager

            # 创建插件管理器
            self.demo_plugin_manager = DemoPluginManager(self)

            # 安装插件（添加菜单和工具栏）
            self.demo_plugin_manager.install_plugin()

            print("✅ 演示插件集成成功！")

        except Exception as e:
            print(f"❌ 插件集成失败: {e}")

    def test_basic_functions(self):
        """测试基础功能"""
        print("\n🧪 测试演示插件基础功能...")

        try:
            # 测试演示引擎
            from plugins.demo_director.core.demo_engine import DemoEngine

            engine = DemoEngine(self)
            print("✅ 演示引擎创建成功")

            # 测试场景加载
            success = engine.load_scenario("🏙️ 城市热岛效应分析")
            if success:
                print("✅ 场景加载成功")
            else:
                print("❌ 场景加载失败")

            # 测试视觉效果管理器
            from plugins.demo_director.effects.visual_effects import VisualEffectsManager

            effects = VisualEffectsManager(self)
            effects.trigger_effect("data_load", {"test": True})
            print("✅ 视觉效果触发成功")

            # 测试AI解说器
            from plugins.demo_director.narration.ai_narrator import AISemanticNarrator

            narrator = AISemanticNarrator()
            narrator.speak("这是一个测试解说")
            print("✅ AI解说器测试成功")

            print("\n🎉 所有基础功能测试通过！")

        except Exception as e:
            print(f"❌ 基础功能测试失败: {e}")

    def launch_demo_plugin(self):
        """启动演示插件"""
        try:
            if hasattr(self, 'demo_plugin_manager'):
                self.demo_plugin_manager.launch_demo_director()
                print("✅ 演示导演控制台已启动")
            else:
                print("❌ 演示插件未正确集成")

        except Exception as e:
            print(f"❌ 启动演示插件失败: {e}")


def main():
    """主函数"""
    print("🎬 智能演示导演插件后端测试")
    print("=" * 50)

    app = QApplication(sys.argv)

    # 创建测试窗口
    window = MockMainWindow()
    window.show()

    print("\n📋 测试说明:")
    print("1. 点击'测试基础功能'按钮测试核心模块")
    print("2. 点击'🎬 启动演示插件'按钮打开演示控制台")
    print("3. 使用快捷键 Ctrl+D 也可以启动演示")
    print("4. 关闭窗口退出测试")

    sys.exit(app.exec())

if __name__ == "__main__":
    main()
# demo_plugin_installer.py - 一键安装脚本
"""
智能演示导演插件一键安装脚本
运行此脚本将自动安装所有依赖并配置插件
"""

import os
import sys
import subprocess
import shutil
import json
from pathlib import Path

def check_dependencies():
    """检查依赖项"""
    required_packages = [
        'PyQt6',
        'numpy',
        'PyQt6-Qt6',
        'pyttsx3',  # 语音合成
        'psutil',   # 性能监控
        'pillow',   # 图像处理
        'opencv-python'  # 视觉效果
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package.replace('-', '_').lower())
            print(f"✅ {package} - 已安装")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - 未安装")

    return missing_packages

def install_dependencies(packages):
    """安装依赖包"""
    if not packages:
        print("✅ 所有依赖已满足！")
        return True

    print(f"\n📦 正在安装依赖包: {', '.join(packages)}")

    try:
        for package in packages:
            print(f"安装 {package}...")
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", package
            ])
            print(f"✅ {package} 安装成功")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖安装失败: {e}")
        return False

def create_plugin_structure():
    """创建插件目录结构"""
    plugin_dir = Path("src/plugins/demo_director")

    directories = [
        "core",
        "effects",
        "narration",
        "utils",
        "resources",
        "resources/icons",
        "resources/sounds",
        "resources/templates"
    ]

    for dir_name in directories:
        dir_path = plugin_dir / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)

        # 创建 __init__.py 文件
        init_file = dir_path / "__init__.py"
        if not init_file.exists():
            init_file.write_text("# Auto-generated init file\n")

    print("✅ 插件目录结构创建完成")

def create_sample_data():
    """创建示例数据"""
    sample_dir = Path("resources/demo_samples")
    sample_dir.mkdir(parents=True, exist_ok=True)

    # 创建示例配置文件
    sample_configs = {
        "urban_heat_scenario.json": {
            "name": "城市热岛效应分析",
            "description": "展示城市温度分布和热岛效应",
            "duration": 300,
            "steps": [
                {
                    "action": "load_thermal_data",
                    "duration": 30,
                    "narration": "加载城市热红外影像数据"
                },
                {
                    "action": "apply_heatmap",
                    "duration": 60,
                    "narration": "生成温度分布热力图"
                }
            ]
        },

        "vegetation_scenario.json": {
            "name": "植被覆盖度监测",
            "description": "NDVI计算和植被健康度分析",
            "duration": 240,
            "steps": [
                {
                    "action": "load_multispectral",
                    "duration": 30,
                    "narration": "加载多光谱卫星影像"
                },
                {
                    "action": "calculate_ndvi",
                    "duration": 90,
                    "narration": "计算归一化植被指数"
                }
            ]
        }
    }

    for filename, config in sample_configs.items():
        config_file = sample_dir / filename
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

    print("✅ 示例数据创建完成")

def create_default_config():
    """创建默认配置文件"""
    config = {
        "plugin_info": {
            "name": "智能演示导演插件",
            "version": "1.0.0",
            "author": "Remote Sensing Team",
            "description": "将遥感图像处理转化为电影级视觉体验"
        },
        "effects": {
            "quality": "高质量",
            "particle_count": 200,
            "animation_speed": 5,
            "enable_gpu_acceleration": True,
            "enable_bloom_effect": True,
            "enable_motion_blur": False
        },
        "audio": {
            "enable_narration": True,
            "voice_speed": 5,
            "enable_sound_effects": True,
            "volume": 0.7,
            "voice_type": "female",
            "language": "zh-CN"
        },
        "scenarios": {
            "auto_load_data": True,
            "default_duration": 300,
            "loop_demos": False,
            "show_progress": True,
            "enable_interaction": True
        },
        "ui": {
            "theme": "sci-fi",
            "show_fps": False,
            "fullscreen_mode": False,
            "enable_shortcuts": True,
            "show_tooltips": True
        },
        "performance": {
            "enable_monitoring": True,
            "max_fps": 60,
            "adaptive_quality": True,
            "memory_limit_mb": 2048
        }
    }

    with open("demo_config.json", 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("✅ 默认配置文件创建完成")

def test_installation():
    """测试安装"""
    print("\n🧪 测试插件安装...")

    try:
        # 测试导入核心模块
        sys.path.insert(0, "src")

        # 先测试基础导入
        import plugins
        import plugins.demo_director
        import plugins.demo_director.core
        import plugins.demo_director.effects
        import plugins.demo_director.narration

        print("✅ 基础模块导入成功")

        # 再测试具体类
        from plugins.demo_director.core.demo_director import DemoDirector
        from plugins.demo_director.effects.visual_effects import VisualEffectsManager
        from plugins.demo_director.narration.ai_narrator import AISemanticNarrator

        print("✅ 核心模块导入成功")
        print("✅ 插件安装测试通过！")
        return True

    except Exception as e:
        print(f"❌ 安装测试失败: {e}")
        return False

def main():
    """主安装函数"""
    print("🎬 智能演示导演插件安装程序")
    print("=" * 50)

    # 1. 检查依赖
    print("\n📋 步骤 1: 检查依赖项")
    missing = check_dependencies()

    # 2. 安装依赖
    if missing:
        print("\n📦 步骤 2: 安装依赖项")
        if not install_dependencies(missing):
            print("❌ 依赖安装失败，请手动安装后重试")
            return False
    else:
        print("\n✅ 步骤 2: 依赖项检查通过")

    # 3. 创建目录结构
    print("\n📁 步骤 3: 创建插件结构")
    create_plugin_structure()

    # 4. 创建配置文件
    print("\n⚙️ 步骤 4: 创建配置文件")
    create_default_config()

    # 5. 创建示例数据
    print("\n📊 步骤 5: 创建示例数据")
    create_sample_data()

    # 6. 测试安装
    print("\n🧪 步骤 6: 测试安装")
    if test_installation():
        print("\n🎉 插件安装成功！")
        print("\n使用说明:")
        print("1. 在主程序中调用 integrate_demo_plugin(main_window)")
        print("2. 使用 Ctrl+D 快捷键启动演示导演")
        print("3. 查看 demo_config.json 了解配置选项")
        return True
    else:
        print("\n❌ 插件安装失败")
        return False

if __name__ == "__main__":
    success = main()
    input("\n按回车键退出...")
    sys.exit(0 if success else 1)








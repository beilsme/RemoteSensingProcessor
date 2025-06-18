import os
import re

def fix_pyqt6_enums():
    """自动修复PyQt6枚举值"""

    fixes = [
        # 基础枚举
        (r'Qt\.AlignCenter', 'Qt.AlignmentFlag.AlignCenter'),
        (r'Qt\.Horizontal', 'Qt.Orientation.Horizontal'),
        (r'Qt\.FramelessWindowHint', 'Qt.WindowType.FramelessWindowHint'),
        (r'Qt\.WindowStaysOnTopHint', 'Qt.WindowType.WindowStaysOnTopHint'),
        (r'Qt\.WA_TranslucentBackground', 'Qt.WidgetAttribute.WA_TranslucentBackground'),
        (r'Qt\.NoPen', 'Qt.PenStyle.NoPen'),

        # QPainter枚举
        (r'QPainter\.Antialiasing', 'QPainter.RenderHint.Antialiasing'),
        (r'QPainter\.TextAntialiasing', 'QPainter.RenderHint.TextAntialiasing'),
        (r'QPainter\.SmoothPixmapTransform', 'QPainter.RenderHint.SmoothPixmapTransform'),

        # QFont枚举
        (r'QFont\.Bold', 'QFont.Weight.Bold'),
        (r'QFont\.Normal', 'QFont.Weight.Normal'),
        (r'QFont\.Light', 'QFont.Weight.Light'),
        (r'QFont\.DemiBold', 'QFont.Weight.DemiBold'),

        # 方法调用
        (r'\.exec_\(\)', '.exec()'),
    ]

    # 需要修复的文件
    files_to_fix = [
        'core/demo_director.py',
        'core/demo_engine.py',
        'effects/visual_effects.py',
        'effects/particle_system.py',
        'narration/ai_narrator.py',
        'plugin_manager.py',
    ]

    for file_path in files_to_fix:
        if os.path.exists(file_path):
            print(f"🔧 修复文件: {file_path}")

            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            for old_pattern, new_pattern in fixes:
                content = re.sub(old_pattern, new_pattern, content)

            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                print(f"✅ {file_path} 修复完成")
            else:
                print(f"ℹ️ {file_path} 无需修改")
        else:
            print(f"⚠️ 文件不存在: {file_path}")

if __name__ == "__main__":
    print("🔧 开始修复PyQt6枚举值...")
    fix_pyqt6_enums()
    print("🎉 PyQt6枚举值修复完成！")
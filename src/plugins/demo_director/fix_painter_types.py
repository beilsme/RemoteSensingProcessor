import os
import re

def fix_painter_types():
    """修复QPainter相关的类型问题"""

    file_path = 'effects/visual_effects.py'

    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_content = content

    # 修复fillRect中的float参数
    fixes = [
        # 修复fillRect参数类型
        (r'painter\.fillRect\((\d+), ([^,]+), ([^,]+), (\d+),',
         r'painter.fillRect(\1, \2, int(\3), \4,'),

        # 修复drawLine中可能的float参数
        (r'painter\.drawLine\(([^,]+), ([^,]+), ([^,]+), ([^)]+)\)',
         r'painter.drawLine(int(\1), int(\2), int(\3), int(\4))'),

        # 修复drawRect中的float参数
        (r'painter\.drawRect\(([^,]+), ([^,]+), ([^,]+), ([^)]+)\)',
         r'painter.drawRect(int(\1), int(\2), int(\3), int(\4))'),

        # 修复drawEllipse中的float参数
        (r'painter\.drawEllipse\(([^,]+), ([^,]+), ([^,]+), ([^)]+)\)',
         r'painter.drawEllipse(int(\1), int(\2), int(\3), int(\4))'),
    ]

    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content)

    # 手动处理一些特殊情况
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if 'fillRect' in line and 'progress_width' in line:
            # 特殊处理progress_width的情况
            lines[i] = line.replace('progress_width', 'int(progress_width)')
        elif 'drawEllipse' in line and 'particle.size' in line:
            # 特殊处理粒子大小的情况
            lines[i] = re.sub(r'int\(([^)]+)\)', r'int(\1)', line)

    content = '\n'.join(lines)

    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ {file_path} QPainter类型问题修复完成")
    else:
        print(f"ℹ️ {file_path} 无QPainter类型问题需要修复")

if __name__ == "__main__":
    print("🔧 开始修复QPainter类型问题...")
    fix_painter_types()
    print("🎉 QPainter类型问题修复完成！")
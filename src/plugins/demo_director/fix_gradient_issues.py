import os
import re

def fix_gradient_issues():
    """修复渐变相关的类型问题"""

    file_path = 'effects/visual_effects.py'

    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 修复QRadialGradient构造
    fixes = [
        # 修复QRadialGradient
        (r'QRadialGradient\(self\.rect\(\)\.center\(\), (\d+)\)',
         r'QRadialGradient(float(self.rect().center().x()), float(self.rect().center().y()), \1.0)'),

        # 修复其他可能的渐变问题
        (r'QLinearGradient\((\w+)\.center\(\)',
         r'QLinearGradient(QPointF(\1.center()))'),
    ]

    original_content = content
    for pattern, replacement in fixes:
        content = re.sub(pattern, replacement, content)

    # 如果还有其他QRadialGradient问题，手动处理
    if 'QRadialGradient(self.rect().center()' in content:
        content = content.replace(
            'QRadialGradient(self.rect().center(), 200)',
            '''center = self.rect().center()
        gradient = QRadialGradient(float(center.x()), float(center.y()), 200.0)'''
        )

    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ {file_path} 渐变问题修复完成")
    else:
        print(f"ℹ️ {file_path} 无渐变问题需要修复")

if __name__ == "__main__":
    print("🔧 开始修复渐变类型问题...")
    fix_gradient_issues()
    print("🎉 渐变问题修复完成！")
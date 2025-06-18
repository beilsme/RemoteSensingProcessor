# plugin_integration_guide.md
"""
# 🎬 智能演示导演插件集成指南

## 快速开始

### 1. 安装插件
```bash
python demo_plugin_installer.py
```

### 2. 在主程序中集成
```python
from plugins.demo_director.plugin_manager import integrate_demo_plugin

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # ... 其他初始化代码 ...
        
        # 集成演示插件
        integrate_demo_plugin(self)
```

### 3. 使用演示功能
- 快捷键 `Ctrl+D` 启动演示导演
- 或点击工具栏的 🎬 按钮
- 选择演示场景并开始

## 自定义演示场景

### 创建新场景
1. 在 `demo_scenarios_config.json` 中添加场景定义
2. 实现对应的动作处理函数
3. 准备示例数据文件

### 场景配置示例
```json
{
  "my_scenario": {
    "name": "我的自定义场景",
    "timeline": [
      {
        "action": "my_custom_action",
        "duration": 60,
        "narration": "这是自定义解说"
      }
    ]
  }
}
```

## API参考

### DemoEngine类
- `load_scenario(name)`: 加载演示场景
- `start_demo()`: 开始演示
- `pause()`: 暂停演示
- `stop()`: 停止演示

### VisualEffectsManager类
- `trigger_effect(name, params)`: 触发视觉效果
- `stop_all_effects()`: 停止所有效果

### 自定义效果
继承 `BaseEffect` 类创建自定义视觉效果：

```python
class MyCustomEffect(BaseEffect):
    def animate_frame(self):
        # 实现动画逻辑
        pass
        
    def paintEvent(self, event):
        # 实现绘制逻辑
        pass
```

## 性能优化

### 建议设置
- 高性能机器：特效强度 8-10，粒子数量 300-500
- 中等性能：特效强度 5-7，粒子数量 150-250
- 低性能设备：特效强度 3-5，粒子数量 50-100

### 监控性能
```python
from plugins.demo_director.utils.performance_monitor import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start_monitoring()
# 获取性能数据
monitor.performance_update.connect(self.on_performance_update)
```

## 故障排除

### 常见问题
1. **语音不工作**: 安装 `pyttsx3` 并检查系统音频
2. **特效卡顿**: 降低粒子数量和特效强度
3. **场景加载失败**: 检查示例数据文件路径

### 调试模式
```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)
```
"""
# src/plugins/demo_director/__init__.py
"""
智能演示导演插件
将遥感图像处理转化为电影级视觉体验
"""

from .core.demo_engine import DemoEngine
from .core.demo_director import DemoDirector
from .effects.visual_effects import VisualEffectsManager
from .narration.ai_narrator import AISemanticNarrator

__version__ = "1.0.0"
__author__ = "Remote Sensing Team"
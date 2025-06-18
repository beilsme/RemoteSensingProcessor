# src/plugins/demo_director/utils/demo_config.py
import json
import os
from typing import Dict, Any, List

class DemoConfig:
    """演示配置管理"""

    def __init__(self):
        self.config_file = "demo_config.json"
        self.default_config = self.get_default_config()
        self.config = self.load_config()

    def get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            "effects": {
                "quality": "高质量",
                "particle_count": 200,
                "animation_speed": 5,
                "enable_gpu_acceleration": True
            },
            "audio": {
                "enable_narration": True,
                "voice_speed": 5,
                "enable_sound_effects": True,
                "volume": 0.7
            },
            "scenarios": {
                "auto_load_data": True,
                "default_duration": 300,  # 5分钟
                "loop_demos": False
            },
            "ui": {
                "theme": "sci-fi",
                "show_fps": False,
                "fullscreen_mode": False
            }
        }

    def load_config(self) -> Dict[str, Any]:
        """加载配置"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"配置加载失败，使用默认配置: {e}")

        return self.default_config.copy()

    def save_config(self):
        """保存配置"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"配置保存失败: {e}")

    def get(self, key_path: str, default=None):
        """获取配置值（支持点号路径）"""
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value):
        """设置配置值（支持点号路径）"""
        keys = key_path.split('.')
        config = self.config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value
        
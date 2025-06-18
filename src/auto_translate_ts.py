# 文件: auto_translate_ts.py
# 功能: 自动翻译 Qt .ts 文件中未翻译的英文 -> 中文
# 作者: 孟诣楠
# 版本: 1.0.0
# 最近更新: 2025-06-18

import time
from xml.etree import ElementTree as ET
from googletrans import Translator

def auto_translate_ts(ts_path: str):
    print(f"🚀 开始翻译：{ts_path}")
    translator = Translator()

    # 加载 ts 文件
    tree = ET.parse(ts_path)
    root = tree.getroot()
    changed_count = 0

    for context in root.findall("context"):
        for message in context.findall("message"):
            source = message.find("source")
            translation = message.find("translation")

            if source is None or translation is None:
                continue

            if translation.text and translation.text.strip():
                continue  # 已翻译

            # 翻译
            text = source.text
            try:
                translated = translator.translate(text, src='en', dest='zh-cn').text
                translation.text = translated
                if 'type' in translation.attrib:
                    del translation.attrib['type']
                changed_count += 1
                print(f"✅ {text} → {translated}")
                time.sleep(0.5)  # 防止频率过高被封
            except Exception as e:
                print(f"❌ 翻译失败: {text} -> {e}")
                continue

    tree.write(ts_path, encoding="utf-8")
    print(f"✅ 翻译完成，共翻译 {changed_count} 条。已保存至原文件。")

if __name__ == "__main__":
    ts_file = "/Users/beilsmindex/遥感图像数字处理系统/RemoteSensingProcessor/RemoteSensingProcessor/resources/translations/zh_CN.ts"
    auto_translate_ts(ts_file)
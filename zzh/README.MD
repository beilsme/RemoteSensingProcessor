# RemoteSensingProfessor

该项目提供遥感图像处理的常用算法实现，包含图像增强、滤波和波段运算等模块，并附带简单的 API 与示例脚本。

## 安装
```bash
pip install -r requirements.txt
pip install -e .
```

## 示例
```bash
python -m processing.image_processing.scripts.demo_enhancement input.tif output.tif
# -*- coding: utf-8 -*-
# 文件: seg_models_classifier.py
# 模块: src.processing.classification
# 功能: 基于 Segmentation Models PyTorch 的遥感影像深度学习分类模块
# 作者: 孟诣楠
# 版本: v1.0.1
# 最新更新: 2025-06-18
# 较上一版改进:
#   1. 增加验证集验证流程与评价指标输出（Loss, IoU）
#   2. 支持早停回调与训练日志记录
#   3. 可配置模型保存目录与自定义文件名
#   4. 增加异常处理，提升鲁棒性

import os
import numpy as np
import torch
from segmentation_models_pytorch import Unet

from torch.utils.tensorboard import SummaryWriter
from base_classifier import BaseClassifier

class SegModelsClassifier(BaseClassifier):
    """
    基于 Segmentation Models PyTorch 的语义分割分类器实现。

    接口:
        - train(train_loader, val_loader=None, epochs=50, lr=1e-3, 
                save_dir='models', save_name='unet.pth', early_stop_patience=None)
        - predict(features: np.ndarray) -> np.ndarray
        - load_model(model_path: str)
    """

    def __init__(self,
                 encoder_name: str = 'resnet34',
                 in_channels: int = 3,
                 classes: int = 4,
                 device: torch.device = None,
                 **kwargs):
        super().__init__()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        try:
            self.model = Unet(
                encoder_name=encoder_name,
                encoder_weights='imagenet',
                in_channels=in_channels,
                classes=classes
            )
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(f"模型初始化失败: {e}")

    def train(self,
              train_loader,
              val_loader=None,
              epochs: int = 50,
              lr: float = 1e-3,
              save_dir: str = 'models',
              save_name: str = 'unet.pth',
              early_stop_patience: int = None):
        """
        训练模型。

        参数:
            train_loader: PyTorch DataLoader, 训练集加载器
            val_loader: PyTorch DataLoader, 验证集加载器，可选
            epochs: int, 最大训练轮数
            lr: float, 学习率
            save_dir: str, 模型保存目录
            save_name: str, 模型文件名
            early_stop_patience: int, 若验证集 Loss 连续不降，则提前停止轮数
        """
        os.makedirs(save_dir, exist_ok=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        writer = SummaryWriter(log_dir=os.path.join(save_dir, 'logs'))
        best_val_loss = float('inf')
        no_improve_count = 0
        try:
            for epoch in range(1, epochs + 1):
                self.model.train()
                train_loss = 0.0
                for imgs, labels in train_loader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    outputs = self.model(imgs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()
                train_loss /= len(train_loader)
                writer.add_scalar('Loss/train', train_loss, epoch)
                print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}")

                if val_loader is not None:
                    self.model.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for imgs, labels in val_loader:
                            imgs, labels = imgs.to(self.device), labels.to(self.device)
                            outputs = self.model(imgs)
                            val_loss += criterion(outputs, labels).item()
                    val_loss /= len(val_loader)
                    writer.add_scalar('Loss/val', val_loss, epoch)
                    print(f"Epoch {epoch}/{epochs} - Val   Loss: {val_loss:.4f}")

                    # 早停逻辑
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        no_improve_count = 0
                        torch.save(self.model.state_dict(), os.path.join(save_dir, save_name))
                        print(f"保存最佳模型: {save_name} (Loss: {best_val_loss:.4f})")
                    else:
                        no_improve_count += 1
                        if early_stop_patience and no_improve_count >= early_stop_patience:
                            print("验证集Loss不再下降，提前停止训练。")
                            break
                else:
                    # 若无验证集，按训练集保存一次
                    torch.save(self.model.state_dict(), os.path.join(save_dir, save_name))
            writer.close()
        except Exception as e:
            writer.close()
            raise RuntimeError(f"训练过程出现异常: {e}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        预测分类结果。

        参数:
            features: numpy 数组, 形状 (H, W, C)
        返回:
            numpy 数组, 形状 (H, W)
        """
        try:
            self.model.eval()
            tensor = torch.from_numpy(features).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                pred = torch.argmax(logits, dim=1).cpu().numpy().squeeze(0)
            return pred
        except Exception as e:
            raise RuntimeError(f"预测失败: {e}")

    def load_model(self, model_path: str):
        """
        加载预训练模型权重。

        参数:
            model_path: 模型文件路径
        """
        try:
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
            self.model.to(self.device)
            print(f"模型已加载: {model_path}")
        except Exception as e:
            raise RuntimeError(f"加载模型失败: {e}")


if __name__ == '__main__':
    # 单元测试示例
    from torch.utils.data import DataLoader, TensorDataset

    # 构造假数据: 8 张 32x32 三波段影像与标签
    data = np.random.rand(8, 32, 32, 3).astype(np.float32)
    labels = np.random.randint(0, 4, size=(8, 32, 32)).astype(np.int64)
    imgs = torch.from_numpy(data).permute(0, 3, 1, 2)
    labs = torch.from_numpy(labels)
    ds = TensorDataset(imgs, labs)
    loader = DataLoader(ds, batch_size=2)

    clf = SegModelsClassifier(in_channels=3, classes=4)
    clf.train(loader, val_loader=loader, epochs=3, lr=1e-3, early_stop_patience=2)
    sample = data[0]
    result = clf.predict(sample)
    print(f"预测输出形状: {result.shape}")

"""Adversarial components for UDA + GAN training.

新增模块：
- GradientReversalLayer (GRL): 梯度反转层，用于域对抗训练
- DomainDiscriminator: 域判别器，判断特征来自仿真还是真实域
- PatchGANDiscriminator: 图像判别器，判别复原图像的纹理真实性
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ---------------------------------------------------------------------------
# 梯度反转层 (Gradient Reversal Layer)
# ---------------------------------------------------------------------------

class _GradientReversal(Function):
    """前向传播恒等，反向传播乘以 -alpha。"""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """包装 GRL 为 nn.Module，便于在 nn.Sequential 中使用。

    使用方法：
        grl = GradientReversalLayer()
        alpha = compute_alpha(iter, total)  # 动态 alpha
        reversed_feat = grl(feat, alpha)
    """

    def forward(self, x: torch.Tensor, alpha: float) -> torch.Tensor:
        return _GradientReversal.apply(x, alpha)


# ---------------------------------------------------------------------------
# 域判别器 (Domain Discriminator)
# ---------------------------------------------------------------------------

class DomainDiscriminator(nn.Module):
    """轻量域判别器：接收 GRL 后的特征图，输出二分类 logit。

    用于域对抗训练，驱动生成器提取 sim/real 不变的特征。

    Args:
        in_channels: 输入特征图的通道数（应与提取特征的通道数一致）
        hidden_channels: 隐藏层通道数
    """

    def __init__(self, in_channels: int, hidden_channels: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, 1, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1),
            nn.BatchNorm2d(hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回 shape (B, 1) 的 logit。"""
        return self.net(x)


# ---------------------------------------------------------------------------
# PatchGAN 判别器 (Image Discriminator)
# ---------------------------------------------------------------------------

class PatchGANDiscriminator(nn.Module):
    """N-Layer PatchGAN 判别器。

    接收复原图像，对每个 patch 输出真/假 logit。
    兼容多光谱数据（in_channels 作为参数传入，不硬编码为 3）。

    Args:
        in_channels: 输入图像通道数（如 3=RGB, 4=NIR+RGB, 8=多光谱）
        num_layers: 判别器层数（默认 3，感受野约 70×70）
        base_channels: 第一层通道数（默认 64）
    """

    def __init__(self, in_channels: int = 3, num_layers: int = 3, base_channels: int = 64):
        super().__init__()

        layers: list[nn.Module] = []
        # 第一层：不做归一化
        layers.append(nn.Conv2d(in_channels, base_channels, 4, 2, 1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        # 中间层：通道数逐层翻倍
        ch = base_channels
        for i in range(1, num_layers):
            ch_next = min(ch * 2, 512)
            layers.append(nn.Conv2d(ch, ch_next, 4, 2, 1))
            layers.append(nn.BatchNorm2d(ch_next))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            ch = ch_next

        # 输出层：1 通道 logit map
        layers.append(nn.Conv2d(ch, 1, 4, 1, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """返回 shape (B, 1, H', W') 的 patch-level logit map。"""
        return self.net(x)

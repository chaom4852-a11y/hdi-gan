from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def shave_border(x: torch.Tensor, border: int) -> torch.Tensor:
    if border <= 0:
        return x
    return x[..., border:-border, border:-border]


def rgb_to_y(x: torch.Tensor) -> torch.Tensor:
    """Convert RGB [0,1] tensor to luma Y."""
    if x.shape[-3] == 1:
        return x
    r, g, b = x[..., 0:1, :, :], x[..., 1:2, :, :], x[..., 2:3, :, :]
    return 0.257 * r + 0.504 * g + 0.098 * b + 16.0 / 255.0


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, border: int = 0, y_channel: bool = False) -> float:
    pred = pred.detach().clamp(0, 1)
    target = target.detach().clamp(0, 1)
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred, size=target.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
    if y_channel:
        pred = rgb_to_y(pred)
        target = rgb_to_y(target)
    pred = shave_border(pred, border)
    target = shave_border(target, border)
    mse = torch.mean((pred - target) ** 2).item()
    if mse <= 1e-12:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _gaussian_kernel(window_size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    k = g[:, None] @ g[None, :]
    return k.view(1, 1, window_size, window_size).repeat(channels, 1, 1, 1)


def calculate_ssim(pred: torch.Tensor, target: torch.Tensor, border: int = 0, y_channel: bool = False) -> float:
    pred = pred.detach().clamp(0, 1)
    target = target.detach().clamp(0, 1)
    if pred.shape[-2:] != target.shape[-2:]:
        pred = F.interpolate(pred, size=target.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
    if y_channel:
        pred = rgb_to_y(pred)
        target = rgb_to_y(target)
    pred = shave_border(pred, border)
    target = shave_border(target, border)

    c = pred.shape[1]
    window = _gaussian_kernel(11, 1.5, c, pred.device, pred.dtype)
    mu1 = F.conv2d(pred, window, padding=5, groups=c)
    mu2 = F.conv2d(target, window, padding=5, groups=c)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu12 = mu1 * mu2
    sigma1_sq = F.conv2d(pred * pred, window, padding=5, groups=c) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=5, groups=c) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=5, groups=c) - mu12
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return float(ssim.mean().item())
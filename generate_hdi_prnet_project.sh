#!/usr/bin/env bash
set -euo pipefail

# Unofficial HDI-PRNet reproduction project generator.
# Run:
#   bash generate_hdi_prnet_project.sh
# Then edit options.yaml and start training:
#   python train.py --config options.yaml

mkdir -p models data utils experiments scripts

cat > models/__init__.py <<'PY'
from .hdi_prnet import HDIPRNet, HDIPRNetConfig, HDIPRNetLoss

__all__ = ["HDIPRNet", "HDIPRNetConfig", "HDIPRNetLoss"]
PY

cat > models/hdi_prnet.py <<'PY'
"""
Unofficial PyTorch implementation of HDI-PRNet.

This implementation is based on the network architecture and text description
from the HDI-PRNet paper. Since official code is unavailable, several low-level
choices are faithful approximations rather than exact author code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_channels: int, out_channels: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=bias)


def conv1x1(in_channels: int, out_channels: int, bias: bool = True) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)


class ResidualChannelAttentionBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.body = nn.Sequential(
            conv3x3(channels, channels),
            nn.PReLU(channels),
            conv3x3(channels, channels),
        )
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv1x1(channels, hidden),
            nn.PReLU(hidden),
            conv1x1(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.body(x)
        return x + feat * self.attn(feat)


class RCABStack(nn.Module):
    def __init__(self, channels: int, num_blocks: int):
        super().__init__()
        self.net = nn.Sequential(*[ResidualChannelAttentionBlock(channels) for _ in range(num_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DenoisingModule(nn.Module):
    """Three-scale RCAB encoder-decoder denoising/proximal module."""

    def __init__(self, in_channels: int = 3, channels: Sequence[int] = (120, 140, 160), num_rcab: int = 2):
        super().__init__()
        c1, c2, c3 = channels
        self.head = conv3x3(in_channels, c1)
        self.enc1 = RCABStack(c1, num_rcab)
        self.down12 = conv3x3(c1, c2)
        self.enc2 = RCABStack(c2, num_rcab)
        self.down23 = conv3x3(c2, c3)
        self.enc3 = RCABStack(c3, num_rcab)
        self.up32 = conv3x3(c3, c2)
        self.skip2 = RCABStack(c2, num_rcab)
        self.dec2 = RCABStack(c2, num_rcab)
        self.up21 = conv3x3(c2, c1)
        self.skip1 = RCABStack(c1, num_rcab)
        self.dec1 = RCABStack(c1, num_rcab)
        self.tail = conv3x3(c1, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shallow = self.head(x)
        e1 = self.enc1(shallow)
        e2_in = F.interpolate(e1, scale_factor=0.5, mode="bilinear", align_corners=False)
        e2 = self.enc2(self.down12(e2_in))
        e3_in = F.interpolate(e2, scale_factor=0.5, mode="bilinear", align_corners=False)
        e3 = self.enc3(self.down23(e3_in))

        d2 = F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(self.up32(d2) + self.skip2(e2))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(self.up21(d1) + self.skip1(e1))
        return x + self.tail(d1 + shallow)


class SuperResolutionModule(nn.Module):
    """Conv3 -> bilinear interpolation -> Conv3 SR module."""

    def __init__(self, channels: int = 3, hidden_channels: int = 64, scale: float = 1.0):
        super().__init__()
        self.scale = float(scale)
        self.head = conv3x3(channels, hidden_channels)
        self.act = nn.PReLU(hidden_channels)
        self.tail = conv3x3(hidden_channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.act(self.head(x))
        if self.scale != 1.0:
            feat = F.interpolate(feat, scale_factor=self.scale, mode="bilinear", align_corners=False)
            base = F.interpolate(x, scale_factor=self.scale, mode="bilinear", align_corners=False)
        else:
            base = x
        return base + self.tail(feat)


class ChannelInteractionAggregation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            conv1x1(channels, hidden),
            nn.PReLU(hidden),
            conv1x1(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class SpatialInteractionAggregation(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            conv1x1(channels, channels),
            nn.PReLU(channels),
            conv1x1(channels, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class SpatialBranch(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(conv3x3(channels, channels), nn.PReLU(channels), conv3x3(channels, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FrequencyBranch(nn.Module):
    """RFFT/IRFFT branch. Complex tensor is represented as concatenated real/imag channels."""

    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            conv1x1(channels * 2, channels * 2),
            nn.PReLU(channels * 2),
            conv1x1(channels * 2, channels * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, c, h, w = x.shape
        z = torch.fft.rfft2(x, norm="ortho")
        zi = torch.cat([z.real, z.imag], dim=1)
        zi = self.net(zi)
        real, imag = torch.chunk(zi, 2, dim=1)
        z = torch.complex(real, imag)
        return torch.fft.irfft2(z, s=(h, w), norm="ortho")


class DualDomainExpressionAggregation(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.sia_s = SpatialInteractionAggregation(channels)
        self.cia_s = ChannelInteractionAggregation(channels)
        self.sia_f = SpatialInteractionAggregation(channels)
        self.cia_f = ChannelInteractionAggregation(channels)
        self.fuse = conv1x1(channels * 2, channels)

    def forward(self, s: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        s = self.sia_s(s) + self.cia_s(s)
        f = self.sia_f(f) + self.cia_f(f)
        return self.fuse(torch.cat([s, f], dim=1))


class DualDomainDegradationLearningBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.pre = nn.Sequential(conv3x3(channels, channels), nn.PReLU(channels), conv3x3(channels, channels))
        self.spatial = SpatialBranch(channels)
        self.frequency = FrequencyBranch(channels)
        self.dea = DualDomainExpressionAggregation(channels)
        self.post = conv1x1(channels, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.pre(x)
        res = self.post(self.dea(self.spatial(feat), self.frequency(feat)))
        return x + res


class DeblurringModule(nn.Module):
    """Truncated Neumann-series deblurring module with DDLB terms."""

    def __init__(self, in_channels: int = 3, channels: int = 64, neumann_terms: int = 5):
        super().__init__()
        self.head = conv3x3(in_channels, channels)
        self.blocks = nn.ModuleList([DualDomainDegradationLearningBlock(channels) for _ in range(neumann_terms)])
        self.tail = conv3x3(channels, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.head(x)
        out = feat
        term = feat
        for block in self.blocks:
            term = block(term)
            out = out + term
        return x + self.tail(out)


@dataclass(frozen=True)
class HDIPRNetConfig:
    in_channels: int = 3
    stages: int = 2
    scale: int = 2
    denoise_channels: Tuple[int, int, int] = (120, 140, 160)
    denoise_rcab_blocks: int = 2
    sr_hidden_channels: int = 64
    deblur_channels: int = 64
    neumann_terms: int = 5


class RestorationStage(nn.Module):
    def __init__(self, cfg: HDIPRNetConfig, stage_scale: float):
        super().__init__()
        self.denoise = DenoisingModule(cfg.in_channels, cfg.denoise_channels, cfg.denoise_rcab_blocks)
        self.sr = SuperResolutionModule(cfg.in_channels, cfg.sr_hidden_channels, stage_scale)
        self.deblur = DeblurringModule(cfg.in_channels, cfg.deblur_channels, cfg.neumann_terms)

    def forward(self, x: torch.Tensor):
        g_dn = self.denoise(x)
        g_sr = self.sr(g_dn)
        g_db = self.deblur(g_sr)
        return g_dn, g_sr, g_db


class HDIPRNet(nn.Module):
    def __init__(self, cfg: HDIPRNetConfig):
        super().__init__()
        if cfg.stages < 1:
            raise ValueError("stages must be >= 1")
        if cfg.scale < 1:
            raise ValueError("scale must be >= 1")
        self.cfg = cfg
        stage_scale = float(cfg.scale) ** (1.0 / float(cfg.stages))
        self.stages = nn.ModuleList([RestorationStage(cfg, stage_scale) for _ in range(cfg.stages)])

    def forward(self, x: torch.Tensor):
        h, w = x.shape[-2:]
        target_size = (h * self.cfg.scale, w * self.cfg.scale)
        mids = []
        out = x
        for stage in self.stages:
            g_dn, g_sr, g_db = stage(out)
            mids.append({"denoise": g_dn, "sr": g_sr, "deblur": g_db})
            out = g_db
        if out.shape[-2:] != target_size:
            out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
        return out, mids


class HDIPRNetLoss(nn.Module):
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, final_pred, final_target, intermediates=None, intermediate_targets=None):
        rec_loss = self.mse(final_pred, final_target)
        int_loss = final_pred.new_tensor(0.0)
        if intermediates is not None and intermediate_targets is not None:
            for pred_dict, target_dict in zip(intermediates, intermediate_targets):
                for key, pred in pred_dict.items():
                    target = target_dict.get(key)
                    if target is None:
                        continue
                    if target.shape[-2:] != pred.shape[-2:]:
                        target = F.interpolate(target, size=pred.shape[-2:], mode="bilinear", align_corners=False)
                    int_loss = int_loss + self.mse(pred, target)
        loss = rec_loss + self.alpha * int_loss
        log = {
            "loss_total": float(loss.detach().cpu()),
            "loss_rec": float(rec_loss.detach().cpu()),
            "loss_int": float(int_loss.detach().cpu()),
        }
        return loss, log


if __name__ == "__main__":
    model = HDIPRNet(HDIPRNetConfig(stages=2, scale=2))
    x = torch.randn(1, 3, 32, 32)
    y, mids = model(x)
    print(x.shape, y.shape)
    print([{k: tuple(v.shape) for k, v in m.items()} for m in mids])
PY

cat > data/degradation.py <<'PY'
"""
High-order synthetic degradation pipeline for HDI-PRNet.

The pipeline follows the paper's degradation sequence per order/stage:
    blur -> resize -> noise
and repeats it k times. It is inspired by Real-ESRGAN style synthetic degradation.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class DegradationConfig:
    scale: int = 2
    orders: int = 2
    final_size_multiple: int = 1
    gray_noise_prob: float = 0.4
    sinc_prob: float = 0.1
    second_blur_skip_prob: float = 0.2
    jpeg_prob: float = 0.0
    return_intermediate_targets: bool = True


def _mesh_grid(kernel_size: int) -> Tuple[np.ndarray, np.ndarray]:
    ax = np.arange(kernel_size, dtype=np.float32) - kernel_size // 2
    xx, yy = np.meshgrid(ax, ax)
    return xx, yy


def _normalize_kernel(k: np.ndarray) -> np.ndarray:
    s = float(np.sum(k))
    if abs(s) < 1e-8:
        k[k.shape[0] // 2, k.shape[1] // 2] = 1.0
        return k.astype(np.float32)
    return (k / s).astype(np.float32)


def random_mixed_gaussian_kernel(kernel_size: int, stage: int) -> np.ndarray:
    """Generate isotropic/anisotropic generalized/plateau-like blur kernels.

    The exact official implementation is unavailable, so this function matches
    the probabilities and parameter ranges from the paper approximately.
    """
    p = random.random()
    probs = [0.45, 0.25, 0.12, 0.03, 0.12, 0.03]
    acc = 0.0
    kind = 0
    for i, pr in enumerate(probs):
        acc += pr
        if p <= acc:
            kind = i
            break

    xx, yy = _mesh_grid(kernel_size)
    sigma_max = 1.0 if stage >= 2 else 2.0
    sigma_x = random.uniform(0.1, sigma_max)
    sigma_y = sigma_x if kind in (0, 2, 4) else random.uniform(0.1, sigma_max)
    theta = random.uniform(0, math.pi)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    xr = cos_t * xx + sin_t * yy
    yr = -sin_t * xx + cos_t * yy
    r2 = (xr / max(sigma_x, 1e-6)) ** 2 + (yr / max(sigma_y, 1e-6)) ** 2

    if kind in (2, 3):
        beta = random.uniform(0.5, 4.0)
        k = np.exp(-0.5 * np.power(np.maximum(r2, 0), beta / 2.0))
    elif kind in (4, 5):
        beta = random.uniform(1.0, 2.0)
        k = 1.0 / np.power(1.0 + r2, beta)
    else:
        k = np.exp(-0.5 * r2)
    return _normalize_kernel(k)


def circular_lowpass_kernel(cutoff: float, kernel_size: int) -> np.ndarray:
    """Sinc-like circular low-pass kernel."""
    ax = np.arange(kernel_size) - kernel_size // 2
    xx, yy = np.meshgrid(ax, ax)
    rr = np.sqrt(xx ** 2 + yy ** 2)
    kernel = cutoff * np.where(rr == 0, cutoff, np.sin(cutoff * rr) / (math.pi * rr))
    window = np.outer(np.hamming(kernel_size), np.hamming(kernel_size))
    return _normalize_kernel(kernel * window)


def random_blur_kernel(stage: int) -> np.ndarray:
    n = random.randint(3, 10)
    kernel_size = 2 * n + 1
    if random.random() < 0.1:
        cutoff = random.uniform(math.pi / 3, math.pi)
        return circular_lowpass_kernel(cutoff, kernel_size)
    return random_mixed_gaussian_kernel(kernel_size, stage)


def filter2d(img: torch.Tensor, kernel: np.ndarray) -> torch.Tensor:
    """Apply the same 2D kernel to each image/channel.

    img: float tensor in [0, 1], shape B,C,H,W.
    """
    b, c, _, _ = img.shape
    k = torch.from_numpy(kernel).to(device=img.device, dtype=img.dtype)
    k = k.view(1, 1, k.shape[0], k.shape[1]).repeat(c, 1, 1, 1)
    pad = kernel.shape[0] // 2
    img = F.pad(img, (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(img, k, groups=c)


def random_resize(img: torch.Tensor, stage: int) -> torch.Tensor:
    if stage >= 2:
        probs = [0.3, 0.4, 0.3]  # up, down, keep
        scale_range = (0.8, 1.2)
    else:
        probs = [0.2, 0.7, 0.1]
        scale_range = (0.5, 1.5)

    r = random.random()
    if r < probs[0]:
        sf = random.uniform(1.0, scale_range[1])
    elif r < probs[0] + probs[1]:
        sf = random.uniform(scale_range[0], 1.0)
    else:
        sf = 1.0

    mode = random.choice(["area", "bilinear", "bicubic"])
    h, w = img.shape[-2:]
    nh, nw = max(4, int(round(h * sf))), max(4, int(round(w * sf)))
    if mode == "area":
        return F.interpolate(img, size=(nh, nw), mode=mode)
    return F.interpolate(img, size=(nh, nw), mode=mode, align_corners=False)


def add_gaussian_noise(img: torch.Tensor, stage: int, gray_prob: float = 0.4) -> torch.Tensor:
    sigma_max = 20.0 if stage >= 2 else 25.0
    sigma = random.uniform(1.0, sigma_max) / 255.0
    b, c, h, w = img.shape
    if random.random() < gray_prob:
        noise = torch.randn(b, 1, h, w, device=img.device, dtype=img.dtype).repeat(1, c, 1, 1)
    else:
        noise = torch.randn_like(img)
    return torch.clamp(img + noise * sigma, 0.0, 1.0)


def add_poisson_noise(img: torch.Tensor, stage: int, gray_prob: float = 0.4) -> torch.Tensor:
    scale_max = 2.0 if stage >= 2 else 2.5
    scale = random.uniform(0.05, scale_max)
    vals = 2 ** torch.ceil(torch.log2(torch.tensor(255.0 * scale, device=img.device)))
    if random.random() < gray_prob:
        gray = img.mean(dim=1, keepdim=True)
        noisy = torch.poisson(gray * vals) / vals
        noisy = noisy.repeat(1, img.shape[1], 1, 1)
    else:
        noisy = torch.poisson(img * vals) / vals
    return torch.clamp(noisy, 0.0, 1.0)


def random_noise(img: torch.Tensor, stage: int, gray_prob: float = 0.4) -> torch.Tensor:
    if random.random() < 0.5:
        return add_gaussian_noise(img, stage, gray_prob)
    return add_poisson_noise(img, stage, gray_prob)


def maybe_jpeg(img: torch.Tensor, prob: float = 0.0) -> torch.Tensor:
    """Optional JPEG degradation. Disabled by default because the paper's main text
    describes blur/resize/noise for HDI-PRNet degradation.
    """
    if prob <= 0 or random.random() >= prob:
        return img
    outs = []
    quality = random.randint(60, 95)
    for x in img.detach().cpu():
        arr = (x.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        ok, enc = cv2.imencode(".jpg", arr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
            dec = cv2.cvtColor(dec, cv2.COLOR_BGR2RGB)
            arr = dec
        t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        outs.append(t)
    return torch.stack(outs, 0).to(device=img.device, dtype=img.dtype)


def degrade_once(img: torch.Tensor, stage: int, cfg: DegradationConfig) -> torch.Tensor:
    if not (stage >= 2 and random.random() < cfg.second_blur_skip_prob):
        img = filter2d(img, random_blur_kernel(stage))
    img = random_resize(img, stage)
    img = random_noise(img, stage, cfg.gray_noise_prob)
    img = maybe_jpeg(img, cfg.jpeg_prob)
    return torch.clamp(img, 0.0, 1.0)


def high_order_degrade(hr: torch.Tensor, cfg: DegradationConfig):
    """Create LR from HR with k-order degradation.

    Returns:
        lr: degraded low-resolution image tensor B,C,H/scale,W/scale
        meta: dict with per-order degraded images and intermediate targets

    Intermediate targets are approximated from HR by resizing to the current
    prediction sizes used by progressive stages. This makes deep supervision
    usable even when the exact synthetic clean states are not stored.
    """
    if hr.dim() == 3:
        hr = hr.unsqueeze(0)
    hr = hr.clamp(0, 1)
    b, c, h, w = hr.shape
    x = hr
    degraded_states: List[torch.Tensor] = []

    for order in range(1, cfg.orders + 1):
        x = degrade_once(x, order, cfg)
        degraded_states.append(x)

    target_h, target_w = h // cfg.scale, w // cfg.scale
    target_h = max(cfg.final_size_multiple, target_h)
    target_w = max(cfg.final_size_multiple, target_w)
    lr = F.interpolate(x, size=(target_h, target_w), mode="bicubic", align_corners=False).clamp(0, 1)

    meta: Dict[str, object] = {"degraded_states": degraded_states}
    return lr, meta
PY

cat > data/dataset.py <<'PY'
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .degradation import DegradationConfig, high_order_degrade

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def scan_images(root: str | Path) -> List[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Image root does not exist: {root}")
    files = [p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTENSIONS]
    if not files:
        raise FileNotFoundError(f"No images found under: {root}")
    return sorted(files)


def read_rgb(path: str | Path) -> torch.Tensor:
    arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if arr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    return t


def random_crop(hr: torch.Tensor, crop_size: int) -> torch.Tensor:
    _, h, w = hr.shape
    if h < crop_size or w < crop_size:
        scale = max(crop_size / h, crop_size / w)
        nh, nw = int(round(h * scale + 0.5)), int(round(w * scale + 0.5))
        hr = F.interpolate(hr.unsqueeze(0), size=(nh, nw), mode="bicubic", align_corners=False).squeeze(0).clamp(0, 1)
        _, h, w = hr.shape
    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    return hr[:, top:top + crop_size, left:left + crop_size]


def augment(hr: torch.Tensor) -> torch.Tensor:
    if random.random() < 0.5:
        hr = torch.flip(hr, dims=[2])
    if random.random() < 0.5:
        hr = torch.flip(hr, dims=[1])
    k = random.randint(0, 3)
    if k:
        hr = torch.rot90(hr, k, dims=[1, 2])
    return hr.contiguous()


def build_intermediate_targets(hr: torch.Tensor, lr: torch.Tensor, stages: int, scale: int) -> List[Dict[str, torch.Tensor]]:
    """Approximate intermediate supervision targets.

    The paper uses intermediate losses for module outputs. The true internal
    targets depend on degradation states. For practical reproduction, we use HR
    resized to each module's predicted spatial size. This encourages every module
    to move toward clean reconstruction while preserving progressive sizes.
    """
    if hr.dim() == 3:
        hr_b = hr.unsqueeze(0)
    else:
        hr_b = hr
    h0, w0 = lr.shape[-2:]
    targets = []
    stage_scale = float(scale) ** (1.0 / float(stages))
    cur_h, cur_w = h0, w0
    for _ in range(stages):
        dn_size = (int(round(cur_h)), int(round(cur_w)))
        sr_size = (int(round(cur_h * stage_scale)), int(round(cur_w * stage_scale)))
        db_size = sr_size
        targets.append({
            "denoise": F.interpolate(hr_b, size=dn_size, mode="bicubic", align_corners=False).squeeze(0).clamp(0, 1),
            "sr": F.interpolate(hr_b, size=sr_size, mode="bicubic", align_corners=False).squeeze(0).clamp(0, 1),
            "deblur": F.interpolate(hr_b, size=db_size, mode="bicubic", align_corners=False).squeeze(0).clamp(0, 1),
        })
        cur_h, cur_w = sr_size
    return targets


class RemoteSensingRestorationDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        crop_size: int = 128,
        scale: int = 2,
        degradation_orders: int = 2,
        stages: int = 2,
        training: bool = True,
        limit: Optional[int] = None,
    ):
        self.paths = scan_images(root)
        if limit is not None:
            self.paths = self.paths[:limit]
        self.crop_size = crop_size
        self.training = training
        self.stages = stages
        self.deg_cfg = DegradationConfig(scale=scale, orders=degradation_orders)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        hr = read_rgb(path)
        if self.training:
            hr = random_crop(hr, self.crop_size)
            hr = augment(hr)
        else:
            # Make dimensions divisible by scale for clean metric calculation.
            _, h, w = hr.shape
            h = h - h % self.deg_cfg.scale
            w = w - w % self.deg_cfg.scale
            hr = hr[:, :h, :w]

        lr, _ = high_order_degrade(hr.unsqueeze(0), self.deg_cfg)
        lr = lr.squeeze(0)
        inter_targets = build_intermediate_targets(hr, lr, self.stages, self.deg_cfg.scale)

        return {
            "lr": lr,
            "hr": hr,
            "intermediate_targets": inter_targets,
            "path": str(path),
        }


def collate_fn(batch):
    # Default collate cannot handle list-of-dict targets cleanly in older PyTorch.
    out = {}
    out["lr"] = torch.stack([b["lr"] for b in batch], 0)
    out["hr"] = torch.stack([b["hr"] for b in batch], 0)
    out["path"] = [b["path"] for b in batch]

    stages = len(batch[0]["intermediate_targets"])
    inter = []
    for s in range(stages):
        d = {}
        for key in batch[0]["intermediate_targets"][s].keys():
            d[key] = torch.stack([b["intermediate_targets"][s][key] for b in batch], 0)
        inter.append(d)
    out["intermediate_targets"] = inter
    return out
PY

cat > utils/metrics.py <<'PY'
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
PY

cat > utils/io.py <<'PY'
from __future__ import annotations

import os
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_image_tensor(x: torch.Tensor, path: str | Path):
    path = Path(path)
    ensure_dir(path.parent)
    if x.dim() == 4:
        x = x[0]
    x = x.detach().float().clamp(0, 1).cpu()
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), arr)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
PY

cat > train.py <<'PY'
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import RemoteSensingRestorationDataset, collate_fn
from models import HDIPRNet, HDIPRNetConfig, HDIPRNetLoss
from utils.io import count_parameters, ensure_dir, load_yaml, save_image_tensor, set_random_seed
from utils.metrics import calculate_psnr, calculate_ssim


def build_model(cfg_dict):
    cfg = HDIPRNetConfig(
        in_channels=cfg_dict.get("in_channels", 3),
        stages=cfg_dict.get("stages", 2),
        scale=cfg_dict.get("scale", 2),
        denoise_channels=tuple(cfg_dict.get("denoise_channels", [120, 140, 160])),
        denoise_rcab_blocks=cfg_dict.get("denoise_rcab_blocks", 2),
        sr_hidden_channels=cfg_dict.get("sr_hidden_channels", 64),
        deblur_channels=cfg_dict.get("deblur_channels", 64),
        neumann_terms=cfg_dict.get("neumann_terms", 5),
    )
    return HDIPRNet(cfg), cfg


def adjust_lr(optimizer, base_lr, iteration, milestones, gamma=0.5):
    factor = gamma ** sum(iteration >= m for m in milestones)
    lr = base_lr * factor
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


@torch.no_grad()
def validate(model, loader, device, scale, save_dir=None, max_save=8):
    model.eval()
    psnr_sum = 0.0
    ssim_sum = 0.0
    n = 0
    for batch_idx, batch in enumerate(tqdm(loader, desc="val", leave=False)):
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        pred, _ = model(lr)
        pred = pred.clamp(0, 1)
        psnr_sum += calculate_psnr(pred, hr, border=scale, y_channel=False)
        ssim_sum += calculate_ssim(pred, hr, border=scale, y_channel=False)
        n += 1
        if save_dir is not None and batch_idx < max_save:
            save_image_tensor(pred[0], Path(save_dir) / f"val_{batch_idx:04d}_pred.png")
            save_image_tensor(hr[0], Path(save_dir) / f"val_{batch_idx:04d}_hr.png")
            save_image_tensor(lr[0], Path(save_dir) / f"val_{batch_idx:04d}_lr.png")
    model.train()
    return {"psnr": psnr_sum / max(n, 1), "ssim": ssim_sum / max(n, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="options.yaml")
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    opt = load_yaml(args.config)
    set_random_seed(opt.get("seed", 1234))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    exp_dir = Path(opt.get("experiment_dir", "experiments/hdi_prnet"))
    ckpt_dir = exp_dir / "checkpoints"
    img_dir = exp_dir / "images"
    ensure_dir(ckpt_dir)
    ensure_dir(img_dir)

    model, model_cfg = build_model(opt["model"])
    model = model.to(device)
    print(f"Model params: {count_parameters(model) / 1e6:.2f} M")

    train_set = RemoteSensingRestorationDataset(
        root=opt["data"]["train_root"],
        crop_size=opt["data"].get("crop_size", 128),
        scale=model_cfg.scale,
        degradation_orders=opt["data"].get("degradation_orders", model_cfg.stages),
        stages=model_cfg.stages,
        training=True,
        limit=opt["data"].get("train_limit"),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=opt["train"].get("batch_size", 8),
        shuffle=True,
        num_workers=opt["train"].get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )

    val_loader = None
    if opt["data"].get("val_root"):
        val_set = RemoteSensingRestorationDataset(
            root=opt["data"]["val_root"],
            crop_size=opt["data"].get("val_crop_size", opt["data"].get("crop_size", 128)),
            scale=model_cfg.scale,
            degradation_orders=opt["data"].get("degradation_orders", model_cfg.stages),
            stages=model_cfg.stages,
            training=False,
            limit=opt["data"].get("val_limit"),
        )
        val_loader = DataLoader(
            val_set,
            batch_size=1,
            shuffle=False,
            num_workers=opt["train"].get("num_workers", 4),
            pin_memory=True,
            collate_fn=collate_fn,
        )

    criterion = HDIPRNetLoss(alpha=opt["train"].get("intermediate_loss_weight", 1.0))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=opt["train"].get("lr", 2e-4),
        betas=tuple(opt["train"].get("betas", [0.9, 0.99])),
    )
    scaler = GradScaler(enabled=opt["train"].get("amp", True) and device.type == "cuda")

    start_iter = 0
    best_psnr = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt.get("iter", 0)
        best_psnr = ckpt.get("best_psnr", -1.0)
        print(f"Resumed from {args.resume} at iter {start_iter}")

    total_iters = opt["train"].get("total_iters", 200000)
    milestones = opt["train"].get("milestones", [20000, 120000, 160000, 180000])
    base_lr = opt["train"].get("lr", 2e-4)
    log_every = opt["train"].get("log_every", 100)
    val_every = opt["train"].get("val_every", 5000)
    save_every = opt["train"].get("save_every", 5000)

    iteration = start_iter
    model.train()
    pbar = tqdm(total=total_iters, initial=start_iter, desc="train")
    while iteration < total_iters:
        for batch in train_loader:
            iteration += 1
            if iteration > total_iters:
                break

            lr_now = adjust_lr(optimizer, base_lr, iteration, milestones, gamma=0.5)
            lr_img = batch["lr"].to(device, non_blocking=True)
            hr_img = batch["hr"].to(device, non_blocking=True)
            inter_targets = [{k: v.to(device, non_blocking=True) for k, v in d.items()} for d in batch["intermediate_targets"]]

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=scaler.is_enabled()):
                pred, mids = model(lr_img)
                loss, logs = criterion(pred, hr_img, mids, inter_targets)

            scaler.scale(loss).backward()
            if opt["train"].get("grad_clip", 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), opt["train"]["grad_clip"])
            scaler.step(optimizer)
            scaler.update()

            if iteration % log_every == 0:
                pbar.set_postfix({"loss": f"{logs['loss_total']:.4f}", "rec": f"{logs['loss_rec']:.4f}", "int": f"{logs['loss_int']:.4f}", "lr": f"{lr_now:.2e}"})

            if val_loader is not None and iteration % val_every == 0:
                metrics = validate(model, val_loader, device, model_cfg.scale, save_dir=img_dir / f"iter_{iteration}")
                print(f"\n[iter {iteration}] val PSNR={metrics['psnr']:.4f}, SSIM={metrics['ssim']:.4f}")
                if metrics["psnr"] > best_psnr:
                    best_psnr = metrics["psnr"]
                    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iter": iteration, "best_psnr": best_psnr, "config": opt}, ckpt_dir / "best.pth")

            if iteration % save_every == 0:
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iter": iteration, "best_psnr": best_psnr, "config": opt}, ckpt_dir / f"iter_{iteration}.pth")

            pbar.update(1)

    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iter": iteration, "best_psnr": best_psnr, "config": opt}, ckpt_dir / "latest.pth")
    pbar.close()


if __name__ == "__main__":
    main()
PY

cat > test.py <<'PY'
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import RemoteSensingRestorationDataset, collate_fn, read_rgb, scan_images
from models import HDIPRNet, HDIPRNetConfig
from utils.io import ensure_dir, load_yaml, save_image_tensor
from utils.metrics import calculate_psnr, calculate_ssim


def build_model(cfg_dict):
    cfg = HDIPRNetConfig(
        in_channels=cfg_dict.get("in_channels", 3),
        stages=cfg_dict.get("stages", 2),
        scale=cfg_dict.get("scale", 2),
        denoise_channels=tuple(cfg_dict.get("denoise_channels", [120, 140, 160])),
        denoise_rcab_blocks=cfg_dict.get("denoise_rcab_blocks", 2),
        sr_hidden_channels=cfg_dict.get("sr_hidden_channels", 64),
        deblur_channels=cfg_dict.get("deblur_channels", 64),
        neumann_terms=cfg_dict.get("neumann_terms", 5),
    )
    return HDIPRNet(cfg), cfg


@torch.no_grad()
def test_synthetic(opt, ckpt_path, output_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = build_model(opt["model"])
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device).eval()

    dataset = RemoteSensingRestorationDataset(
        root=opt["data"]["test_root"],
        crop_size=opt["data"].get("crop_size", 128),
        scale=cfg.scale,
        degradation_orders=opt["data"].get("degradation_orders", cfg.stages),
        stages=cfg.stages,
        training=False,
        limit=opt["data"].get("test_limit"),
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=opt["test"].get("num_workers", 2), collate_fn=collate_fn)
    ensure_dir(output_dir)
    psnr_sum, ssim_sum, n = 0.0, 0.0, 0
    for batch in tqdm(loader, desc="test synthetic"):
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)
        pred, mids = model(lr)
        pred = pred.clamp(0, 1)
        psnr = calculate_psnr(pred, hr, border=cfg.scale)
        ssim = calculate_ssim(pred, hr, border=cfg.scale)
        psnr_sum += psnr
        ssim_sum += ssim
        n += 1
        name = Path(batch["path"][0]).stem
        save_image_tensor(pred[0], Path(output_dir) / f"{name}_pred.png")
        if opt["test"].get("save_lr_hr", True):
            save_image_tensor(lr[0], Path(output_dir) / f"{name}_lr.png")
            save_image_tensor(hr[0], Path(output_dir) / f"{name}_hr.png")
    print(f"Average PSNR: {psnr_sum / max(n, 1):.4f}")
    print(f"Average SSIM: {ssim_sum / max(n, 1):.4f}")


@torch.no_grad()
def test_real_folder(opt, ckpt_path, input_dir, output_dir):
    """Run model on existing LR/real remote-sensing images without metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = build_model(opt["model"])
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device).eval()
    ensure_dir(output_dir)
    for path in tqdm(scan_images(input_dir), desc="test real"):
        x = read_rgb(path).unsqueeze(0).to(device)
        pred, _ = model(x)
        save_image_tensor(pred[0], Path(output_dir) / f"{path.stem}_x{cfg.scale}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="options.yaml")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--mode", type=str, default="synthetic", choices=["synthetic", "real"])
    parser.add_argument("--input", type=str, default="")
    parser.add_argument("--output", type=str, default="experiments/hdi_prnet/test_results")
    args = parser.parse_args()
    opt = load_yaml(args.config)
    if args.mode == "synthetic":
        test_synthetic(opt, args.ckpt, args.output)
    else:
        if not args.input:
            raise ValueError("--input is required in real mode")
        test_real_folder(opt, args.ckpt, args.input, args.output)


if __name__ == "__main__":
    main()
PY

cat > options.yaml <<'YAML'
seed: 1234
experiment_dir: experiments/hdi_prnet_x2

model:
  in_channels: 3
  stages: 2              # Paper ablation favors 2-stage high-order restoration.
  scale: 2               # Use 2, 3, or 4.
  denoise_channels: [120, 140, 160]
  denoise_rcab_blocks: 2
  sr_hidden_channels: 64
  deblur_channels: 64
  neumann_terms: 5       # Paper ablation favors 5 Neumann terms.

data:
  train_root: /path/to/AID_or_WHU_Building/train
  val_root: /path/to/WHU-RS19/val
  test_root: /path/to/WHU-RS19/test
  crop_size: 128         # For x2: HR crop 128 gives LR about 64.
  val_crop_size: 256
  degradation_orders: 2
  train_limit: null
  val_limit: null
  test_limit: null

train:
  batch_size: 8          # Paper uses 32 on A100; reduce if memory is insufficient.
  num_workers: 4
  total_iters: 200000
  lr: 0.0002
  betas: [0.9, 0.99]
  milestones: [20000, 120000, 160000, 180000]
  intermediate_loss_weight: 1.0
  amp: true
  grad_clip: 0
  log_every: 100
  val_every: 5000
  save_every: 5000

test:
  num_workers: 2
  save_lr_hr: true
YAML

cat > README.md <<'MD'
# HDI-PRNet Unofficial Reproduction

This is an unofficial PyTorch reproduction of **HDI-PRNet: A Progressive Image Restoration Network for High-order Degradation Imaging in Remote Sensing**.

The official code is not public, so this implementation follows the paper's architecture diagrams and descriptions as closely as possible while making practical assumptions where details are missing.

## Implemented

- Progressive high-order restoration network
- Denoising module with 3-scale RCAB encoder-decoder
- SR module with Conv + bilinear interpolation + Conv
- Deblurring module with truncated Neumann expansion
- Dual-domain degradation learning block using spatial and frequency branches
- CIA/SIA/DEA feature interaction
- Reconstruction loss + intermediate supervision
- High-order synthetic degradation: blur -> resize -> noise, repeated k times
- Training, validation, synthetic testing, and real-image inference scripts

## Install

```bash
conda create -n hdi-prnet python=3.10 -y
conda activate hdi-prnet
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install opencv-python tqdm pyyaml numpy
```

## Prepare data

Place HR remote sensing images in folders such as:

```text
/path/to/AID_or_WHU_Building/train
/path/to/WHU-RS19/val
/path/to/WHU-RS19/test
```

The dataset loader recursively scans common image extensions.

## Train

Edit `options.yaml`, then run:

```bash
python train.py --config options.yaml
```

## Resume

```bash
python train.py --config options.yaml --resume experiments/hdi_prnet_x2/checkpoints/iter_5000.pth
```

## Test on synthetic degraded images

```bash
python test.py --config options.yaml --ckpt experiments/hdi_prnet_x2/checkpoints/best.pth --mode synthetic --output results/synthetic_x2
```

## Test on real LR satellite images

```bash
python test.py --config options.yaml --ckpt experiments/hdi_prnet_x2/checkpoints/best.pth --mode real --input /path/to/real_lr_images --output results/real_x2
```

## Notes on reproducibility

The paper reports training with batch size 32, 200K iterations, Adam betas 0.9/0.99, initial LR 2e-4, milestones 20K/120K/160K/180K, and A100 GPU. This project mirrors those settings in `options.yaml`, but the default batch size is lowered to 8 for easier local testing.

For stronger alignment with the paper:

1. Use AID + WHU Building for training.
2. Use WHU-RS19, DOTA, RSSCN7, UCMerced, and NWPU-RESISC45 for testing.
3. Train separate models for x2, x3, and x4.
4. Increase batch size to 32 if GPU memory allows.
5. Replace the approximate degradation implementation with a BasicSR/Real-ESRGAN-style kernel generator if exact matching is required.
MD

cat > scripts/smoke_test.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
python -m models.hdi_prnet
SH
chmod +x scripts/smoke_test.sh

cat > .gitignore <<'TXT'
__pycache__/
*.pyc
*.pth
*.pt
experiments/
results/
.DS_Store
TXT

echo "HDI-PRNet unofficial project files generated."
echo "Next: edit options.yaml and run python train.py --config options.yaml"

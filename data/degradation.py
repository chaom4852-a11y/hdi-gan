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
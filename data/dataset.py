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
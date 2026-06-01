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

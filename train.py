from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data.dataset import RemoteSensingRestorationDataset, collate_fn
from models import (
    HDIPRNet,
    HDIPRNetConfig,
    HDIPRNetLoss,
    EnhancedRestorationLoss,
    GradientReversalLayer,
    DomainDiscriminator,
    PatchGANDiscriminator,
)
from utils.io import count_parameters, ensure_dir, load_yaml, save_image_tensor, set_random_seed
from utils.metrics import calculate_psnr, calculate_ssim


# ---------------------------------------------------------------------------
# 新增部分：真实退化图像 Dataset（无配对 HR）
# ---------------------------------------------------------------------------

_IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


class RealImageDataset(Dataset):
    """加载真实退化低分辨率图像，不进行合成退化。

    仅用于提供真实域分布的 LR 图像，与仿真域 LR 做域对齐。
    随机裁剪 + 翻转 + 旋转增强，保持与仿真数据集一致的增强策略。
    """

    def __init__(self, root: str | Path, crop_size: int = 64, limit: Optional[int] = None):
        root = Path(root)
        if not root.exists():
            raise FileNotFoundError(f"Real image root does not exist: {root}")
        self.paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in _IMG_EXT)
        if not self.paths:
            raise FileNotFoundError(f"No images found under: {root}")
        if limit is not None:
            self.paths = self.paths[:limit]
        self.crop_size = crop_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        arr = cv2.imread(str(self.paths[idx]), cv2.IMREAD_COLOR)
        if arr is None:
            raise RuntimeError(f"Failed to read image: {self.paths[idx]}")
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0

        # 随机裁剪（与仿真数据集保持一致的增强策略）
        _, h, w = img.shape
        cs = self.crop_size
        if h >= cs and w >= cs:
            top = np.random.randint(0, h - cs + 1)
            left = np.random.randint(0, w - cs + 1)
            img = img[:, top:top + cs, left:left + cs]
        else:
            img = F.interpolate(img.unsqueeze(0), size=(cs, cs), mode="bicubic", align_corners=False).squeeze(0).clamp(0, 1)

        # 随机翻转 + 旋转（与仿真数据集增强策略一致）
        if np.random.random() < 0.5:
            img = torch.flip(img, dims=[2])
        if np.random.random() < 0.5:
            img = torch.flip(img, dims=[1])
        k = np.random.randint(0, 4)
        if k:
            img = torch.rot90(img, k, dims=[1, 2])

        return {"lr": img.contiguous()}


def _real_collate_fn(batch):
    return {"lr": torch.stack([b["lr"] for b in batch], 0)}


# ---------------------------------------------------------------------------
# 模型构建
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 学习率调度
# ---------------------------------------------------------------------------

def adjust_lr(optimizer, base_lr, iteration, milestones, gamma=0.5):
    factor = gamma ** sum(iteration >= m for m in milestones)
    lr = base_lr * factor
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


# ---------------------------------------------------------------------------
# 动态 alpha 计算（DANN 标准公式）
# ---------------------------------------------------------------------------

def compute_alpha(iteration: int, total_iters: int) -> float:
    """DANN 标准 alpha 调度：从 0 单调渐增到接近 1。

    公式: alpha = 2 / (1 + exp(-10 * p)) - 1,  其中 p = iter / total_iters
    早期 alpha 小 → 域判别器先学会区分；
    中后期 alpha 增大 → 生成器开始对抗，特征逐渐对齐。
    """
    p = min(iteration / max(total_iters, 1), 1.0)
    return 2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 主训练入口
# ---------------------------------------------------------------------------

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

    # -----------------------------------------------------------------------
    # 1. 构建生成器（主干网络）
    # -----------------------------------------------------------------------
    model, model_cfg = build_model(opt["model"])
    model = model.to(device)
    print(f"[Generator] params: {count_parameters(model) / 1e6:.2f} M")

    # -----------------------------------------------------------------------
    # 2. 新增部分：构建对抗组件（GRL + 域判别器 + PatchGAN 判别器）
    # -----------------------------------------------------------------------
    grl = GradientReversalLayer()

    # 域判别器：输入通道 = denoise_channels[0]（DenoisingModule tail 之前的中间特征）
    feat_channels = model_cfg.denoise_channels[0]  # 默认 120
    domain_disc = DomainDiscriminator(
        in_channels=feat_channels,
        hidden_channels=opt["train"].get("domain_disc_hidden", 256),
    ).to(device)

    # PatchGAN 判别器：输入通道 = 图像通道数（兼容多光谱）
    patch_disc = PatchGANDiscriminator(
        in_channels=model_cfg.in_channels,
        num_layers=opt["train"].get("patch_gan_layers", 3),
        base_channels=opt["train"].get("patch_gan_base_ch", 64),
    ).to(device)

    print(f"[DomainDisc] params: {count_parameters(domain_disc) / 1e6:.2f} M")
    print(f"[PatchGAN]  params: {count_parameters(patch_disc) / 1e6:.2f} M")

    # -----------------------------------------------------------------------
    # 3. 构建双流 DataLoader
    # -----------------------------------------------------------------------
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

    real_loader = None
    if opt["data"].get("real_root"):
        # 真实图像 Dataset：不做合成退化，crop_size 设为 LR 尺寸
        real_set = RealImageDataset(
            root=opt["data"]["real_root"],
            crop_size=opt["data"].get("crop_size", 128) // model_cfg.scale,
            limit=opt["data"].get("train_limit"),
        )
        real_loader = DataLoader(
            real_set,
            batch_size=opt["train"].get("batch_size", 8),
            shuffle=True,
            num_workers=opt["train"].get("num_workers", 4),
            pin_memory=True,
            drop_last=True,
            collate_fn=_real_collate_fn,
        )
        print(f"[Data] Sim dataset: {len(train_set)} images, Real dataset: {len(real_set)} images")
    else:
        print(f"[Data] Sim dataset: {len(train_set)} images, Real dataset: DISABLED (no real_root)")

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

    # -----------------------------------------------------------------------
    # 4. 新增部分：构建联合损失函数（替代原有 HDIPRNetLoss）
    # -----------------------------------------------------------------------
    criterion = EnhancedRestorationLoss(
        domain_discriminator=domain_disc,
        patch_discriminator=patch_disc,
        grl=grl,
        lambda_rec=opt["train"].get("lambda_rec", 1.0),
        lambda_int=opt["train"].get("lambda_int", 1.0),
        lambda_dom=opt["train"].get("lambda_dom", 0.01),
        lambda_gan=opt["train"].get("lambda_gan", 0.001),
    )

    # -----------------------------------------------------------------------
    # 5. 新增部分：三组优化器（Generator + DomainDisc + PatchGAN）
    # -----------------------------------------------------------------------
    base_lr = opt["train"].get("lr", 2e-4)
    lr_disc = opt["train"].get("lr_disc", 2e-4)
    betas = tuple(opt["train"].get("betas", [0.9, 0.99]))

    # 生成器优化器：主干网络参数 + GRL（无可学习参数）+ criterion 内嵌判别器不在此优化
    optimizer_G = torch.optim.Adam(model.parameters(), lr=base_lr, betas=betas)

    # 判别器优化器：域判别器 + PatchGAN 判别器（参数分组便于分别记录）
    optimizer_D = torch.optim.Adam([
        {"params": domain_disc.parameters(), "lr": lr_disc, "betas": betas},
        {"params": patch_disc.parameters(), "lr": lr_disc, "betas": betas},
    ])

    # 混合精度 GradScaler（生成器和判别器各一个）
    amp_enabled = opt["train"].get("amp", True) and device.type == "cuda"
    scaler_G = GradScaler(enabled=amp_enabled)
    scaler_D = GradScaler(enabled=amp_enabled)

    # -----------------------------------------------------------------------
    # 6. 断点续训
    # -----------------------------------------------------------------------
    start_iter = 0
    best_psnr = -1.0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        optimizer_G.load_state_dict(ckpt["optimizer_G"])
        if "optimizer_D" in ckpt:
            optimizer_D.load_state_dict(ckpt["optimizer_D"])
        if "domain_disc" in ckpt:
            domain_disc.load_state_dict(ckpt["domain_disc"])
        if "patch_disc" in ckpt:
            patch_disc.load_state_dict(ckpt["patch_disc"])
        start_iter = ckpt.get("iter", 0)
        best_psnr = ckpt.get("best_psnr", -1.0)
        print(f"Resumed from {args.resume} at iter {start_iter}")

    # -----------------------------------------------------------------------
    # 7. 新增部分：双流交替优化训练循环
    # -----------------------------------------------------------------------
    total_iters = opt["train"].get("total_iters", 200000)
    milestones = opt["train"].get("milestones", [20000, 120000, 160000, 180000])
    log_every = opt["train"].get("log_every", 100)
    val_every = opt["train"].get("val_every", 5000)
    save_every = opt["train"].get("save_every", 5000)

    iteration = start_iter
    model.train()
    domain_disc.train()
    patch_disc.train()

    pbar = tqdm(total=total_iters, initial=start_iter, desc="train")
    sim_iter = iter(train_loader)
    real_iter = iter(real_loader) if real_loader is not None else None

    while iteration < total_iters:
        # ======== 采样双流数据 ========
        # 仿真流：配对的 (lr, hr, intermediate_targets)
        try:
            sim_batch = next(sim_iter)
        except StopIteration:
            sim_iter = iter(train_loader)
            sim_batch = next(sim_iter)

        # 真实流：仅 lr（无配对 HR）
        if real_iter is not None:
            try:
                real_batch = next(real_iter)
            except StopIteration:
                real_iter = iter(real_loader)
                real_batch = next(real_iter)
        else:
            # 无真实数据时退化为纯监督训练（仅用仿真数据）
            real_batch = sim_batch

        iteration += 1
        if iteration > total_iters:
            break

        # 动态 alpha（DANN 标准调度）
        alpha = compute_alpha(iteration, total_iters)

        # 学习率调度
        lr_now = adjust_lr(optimizer_G, base_lr, iteration, milestones, gamma=0.5)
        adjust_lr(optimizer_D, lr_disc, iteration, milestones, gamma=0.5)

        # 数据搬移到 GPU
        lr_sim = sim_batch["lr"].to(device, non_blocking=True)
        hr_sim = sim_batch["hr"].to(device, non_blocking=True)
        inter_targets = [
            {k: v.to(device, non_blocking=True) for k, v in d.items()}
            for d in sim_batch["intermediate_targets"]
        ]
        lr_real = real_batch["lr"].to(device, non_blocking=True)

        # ==================================================================
        # 步骤 A：更新生成器（固定判别器）
        # ==================================================================
        optimizer_G.zero_grad(set_to_none=True)
        with autocast(enabled=scaler_G.is_enabled()):
            # 仿真数据前向：输出 + 中间输出 + 域对齐特征
            pred_sim, mids, feat_sim = model(lr_sim, return_feat=True)
            # 真实数据前向：输出 + 特征（不需要中间输出）
            pred_real, _, feat_real = model(lr_real, return_feat=True)

            loss_G, logs = criterion(
                pred_sim=pred_sim,
                hr_sim=hr_sim,
                feat_sim=feat_sim,
                pred_real=pred_real,
                feat_real=feat_real,
                alpha=alpha,
                intermediates=mids,
                intermediate_targets=inter_targets,
            )

        scaler_G.scale(loss_G).backward()
        if opt["train"].get("grad_clip", 0) > 0:
            scaler_G.unscale_(optimizer_G)
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt["train"]["grad_clip"])
        scaler_G.step(optimizer_G)
        scaler_G.update()

        # ==================================================================
        # 步骤 B：更新判别器（固定生成器，使用 .detach()）
        # ==================================================================
        optimizer_D.zero_grad(set_to_none=True)
        with autocast(enabled=scaler_D.is_enabled()):
            # 域判别器 loss（detach 特征，仅更新判别器权重）
            # 仿真特征 → 标签 0（判为仿真域）
            dom_logit_sim_D = domain_disc(grl(feat_sim.detach(), alpha))
            # 真实特征 → 标签 1（判为真实域）
            dom_logit_real_D = domain_disc(grl(feat_real.detach(), alpha))
            loss_D_dom = nn.functional.binary_cross_entropy_with_logits(dom_logit_sim_D, torch.zeros_like(dom_logit_sim_D)) + \
                         nn.functional.binary_cross_entropy_with_logits(dom_logit_real_D, torch.ones_like(dom_logit_real_D))

            # PatchGAN 判别器 loss（detach 预测图，仅更新判别器权重）
            # 真实图像复原结果 → 标签 1
            patch_logit_hr = patch_disc(hr_sim.detach())
            # 仿真复原结果 → 标签 0
            patch_logit_sim_D = patch_disc(pred_sim.detach())
            # 真实域复原结果 → 标签 0
            patch_logit_real_D = patch_disc(pred_real.detach())
            loss_D_patch = nn.functional.binary_cross_entropy_with_logits(patch_logit_hr, torch.ones_like(patch_logit_hr)) + \
                           nn.functional.binary_cross_entropy_with_logits(patch_logit_sim_D, torch.zeros_like(patch_logit_sim_D)) + \
                           nn.functional.binary_cross_entropy_with_logits(patch_logit_real_D, torch.zeros_like(patch_logit_real_D))

            loss_D = loss_D_dom + loss_D_patch

        scaler_D.scale(loss_D).backward()
        scaler_D.step(optimizer_D)
        scaler_D.update()

        # ==================================================================
        # 日志 & 验证 & 保存
        # ==================================================================
        if iteration % log_every == 0:
            logs_D = {
                "D_dom": float(loss_D_dom.detach().cpu()),
                "D_patch": float(loss_D_patch.detach().cpu()),
            }
            pbar.set_postfix({
                "G": f"{float(logs['loss_total'].cpu()):.4f}",
                "rec": f"{float(logs['loss_rec'].cpu()):.4f}",
                "dom": f"{float(logs['loss_dom'].cpu()):.4f}",
                "gan": f"{float(logs['loss_gan'].cpu()):.4f}",
                "D": f"{float(loss_D.detach().cpu()):.4f}",
                "α": f"{alpha:.3f}",
                "lr": f"{lr_now:.2e}",
            })

        if val_loader is not None and iteration % val_every == 0:
            metrics = validate(model, val_loader, device, model_cfg.scale, save_dir=img_dir / f"iter_{iteration}")
            print(f"\n[iter {iteration}] val PSNR={metrics['psnr']:.4f}, SSIM={metrics['ssim']:.4f}")
            if metrics["psnr"] > best_psnr:
                best_psnr = metrics["psnr"]
                torch.save({
                    "model": model.state_dict(),
                    "domain_disc": domain_disc.state_dict(),
                    "patch_disc": patch_disc.state_dict(),
                    "optimizer_G": optimizer_G.state_dict(),
                    "optimizer_D": optimizer_D.state_dict(),
                    "iter": iteration,
                    "best_psnr": best_psnr,
                    "config": opt,
                }, ckpt_dir / "best.pth")

        if iteration % save_every == 0:
            torch.save({
                "model": model.state_dict(),
                "domain_disc": domain_disc.state_dict(),
                "patch_disc": patch_disc.state_dict(),
                "optimizer_G": optimizer_G.state_dict(),
                "optimizer_D": optimizer_D.state_dict(),
                "iter": iteration,
                "best_psnr": best_psnr,
                "config": opt,
            }, ckpt_dir / f"iter_{iteration}.pth")

        pbar.update(1)

    # 保存最终 checkpoint
    torch.save({
        "model": model.state_dict(),
        "domain_disc": domain_disc.state_dict(),
        "patch_disc": patch_disc.state_dict(),
        "optimizer_G": optimizer_G.state_dict(),
        "optimizer_D": optimizer_D.state_dict(),
        "iter": iteration,
        "best_psnr": best_psnr,
        "config": opt,
    }, ckpt_dir / "latest.pth")
    pbar.close()


if __name__ == "__main__":
    main()

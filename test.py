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
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

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        """前向传播。

        Args:
            x: 输入 (B, in_channels, H, W)
            return_feat: 若为 True，额外返回 tail 之前的中间特征（通道数 = channels[0]）
        Returns:
            output: 去噪结果 (B, in_channels, H, W)
            feat (可选): tail 之前的 decoder 输出 (B, channels[0], H, W)
        """
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
        feat = d1 + shallow  # tail 之前的中间特征，通道数 = channels[0]
        if return_feat:
            return x + self.tail(feat), feat
        return x + self.tail(feat)


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

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        if return_feat:
            g_dn, denoise_feat = self.denoise(x, return_feat=True)
        else:
            g_dn = self.denoise(x)
            denoise_feat = None
        g_sr = self.sr(g_dn)
        g_db = self.deblur(g_sr)
        if return_feat:
            return g_dn, g_sr, g_db, denoise_feat
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

    def forward(self, x: torch.Tensor, return_feat: bool = False):
        """
        Args:
            x: 输入低分辨率图像 (B, C, H, W)
            return_feat: 若为 True，额外返回第一个 stage 去噪模块的中间特征用于域对齐
        Returns:
            out: 复原结果 (B, C, H*scale, W*scale)
            mids: 各阶段中间输出列表
            feat (可选): 去噪模块 tail 之前的中间特征 (B, denoise_channels[0], H, W)
        """
        h, w = x.shape[-2:]
        target_size = (h * self.cfg.scale, w * self.cfg.scale)
        mids = []
        feat = None
        out = x
        for i, stage in enumerate(self.stages):
            if i == 0 and return_feat:
                g_dn, g_sr, g_db, denoise_feat = stage(out, return_feat=True)
                feat = denoise_feat
            else:
                g_dn, g_sr, g_db = stage(out)
            mids.append({"denoise": g_dn, "sr": g_sr, "deblur": g_db})
            out = g_db
        if out.shape[-2:] != target_size:
            out = F.interpolate(out, size=target_size, mode="bilinear", align_corners=False)
        if return_feat:
            return out, mids, feat
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


class EnhancedRestorationLoss(nn.Module):
    """新增部分：UDA + GAN 联合损失函数。

    包含四项损失：
    1. 重建损失 (L_recon): 仅在仿真数据上计算的 L1 Loss
    2. 中间监督损失 (L_int): 各阶段子模块输出的 L1 Loss（保留以稳定训练）
    3. 域对齐损失 (L_dom): 基于 GRL + 域判别器的 BCE Loss
    4. 对抗损失 (L_gan): 基于 PatchGAN 判别器的生成器 BCE Loss

    注意：不使用 VGG 感知损失，因为需兼容多光谱数据。
    """

    def __init__(
        self,
        domain_discriminator: nn.Module,
        patch_discriminator: nn.Module,
        grl: nn.Module,
        lambda_rec: float = 1.0,
        lambda_int: float = 1.0,
        lambda_dom: float = 0.01,
        lambda_gan: float = 0.001,
    ):
        super().__init__()
        self.domain_disc = domain_discriminator
        self.patch_disc = patch_discriminator
        self.grl = grl
        self.lambda_rec = lambda_rec
        self.lambda_int = lambda_int
        self.lambda_dom = lambda_dom
        self.lambda_gan = lambda_gan
        self.l1 = nn.L1Loss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(
        self,
        pred_sim: torch.Tensor,
        hr_sim: torch.Tensor,
        feat_sim: torch.Tensor,
        pred_real: torch.Tensor,
        feat_real: torch.Tensor,
        alpha: float,
        intermediates=None,
        intermediate_targets=None,
    ):
        """
        Args:
            pred_sim: 仿真数据的复原结果 (B, C, H', W')
            hr_sim: 仿真数据的 HR ground truth (B, C, H', W')
            feat_sim: 仿真数据的中间特征 (B, C', H'', W'')，用于域对齐
            pred_real: 真实数据的复原结果 (B, C, H', W')
            feat_real: 真实数据的中间特征 (B, C', H'', W'')，用于域对齐
            alpha: GRL 的动态权重（随训练进度从 0 渐增到 ~1）
            intermediates: 仿真数据的各阶段中间输出
            intermediate_targets: 仿真数据的各阶段中间监督目标
        Returns:
            loss: 总损失标量
            log: 各项损失的字典（用于日志记录）
        """
        # --- 1. 重建损失：仅在仿真数据上计算 ---
        loss_rec = self.l1(pred_sim, hr_sim)

        # --- 2. 中间监督损失：保留原有逻辑，用 L1 替代 MSE ---
        loss_int = pred_sim.new_tensor(0.0)
        if intermediates is not None and intermediate_targets is not None:
            for pred_dict, target_dict in zip(intermediates, intermediate_targets):
                for key, pred in pred_dict.items():
                    target = target_dict.get(key)
                    if target is None:
                        continue
                    if target.shape[-2:] != pred.shape[-2:]:
                        target = F.interpolate(
                            target, size=pred.shape[-2:], mode="bilinear", align_corners=False
                        )
                    loss_int = loss_int + self.l1(pred, target)

        # --- 3. 域对齐损失：GRL + 域判别器 ---
        # GRL 前向恒等、反向反转梯度，驱动生成器提取 sim/real 不变的特征
        # sim 特征经 GRL 后，域判别器应预测为 1（真实域）
        feat_sim_rev = self.grl(feat_sim, alpha)
        dom_logit_sim = self.domain_disc(feat_sim_rev)
        # real 特征经 GRL 后，域判别器应预测为 0（仿真域）
        feat_real_rev = self.grl(feat_real, alpha)
        dom_logit_real = self.domain_disc(feat_real_rev)
        # 生成器视角：希望判别器分不清 → sim 和 real 都被判为"真域"
        loss_dom = self.bce(dom_logit_sim, torch.ones_like(dom_logit_sim)) + \
                   self.bce(dom_logit_real, torch.zeros_like(dom_logit_real))

        # --- 4. 对抗损失：PatchGAN 生成器 loss ---
        # 生成器希望判别器认为 sim 和 real 的复原结果都为真
        gan_logit_sim = self.patch_disc(pred_sim)
        gan_logit_real = self.patch_disc(pred_real)
        loss_gan = self.bce(gan_logit_sim, torch.ones_like(gan_logit_sim)) + \
                   self.bce(gan_logit_real, torch.ones_like(gan_logit_real))

        # --- 加权求和 ---
        loss_total = (
            self.lambda_rec * loss_rec
            + self.lambda_int * loss_int
            + self.lambda_dom * loss_dom
            + self.lambda_gan * loss_gan
        )

        log = {
            "loss_total": loss_total.detach(),
            "loss_rec": loss_rec.detach(),
            "loss_int": loss_int.detach(),
            "loss_dom": loss_dom.detach(),
            "loss_gan": loss_gan.detach(),
        }
        return loss_total, log


if __name__ == "__main__":
    model = HDIPRNet(HDIPRNetConfig(stages=2, scale=2))
    x = torch.randn(1, 3, 32, 32)
    y, mids = model(x)
    print(x.shape, y.shape)
    print([{k: tuple(v.shape) for k, v in m.items()} for m in mids])
import argparse
import random
import shutil
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import transforms

# gcs
from compressai.datasets import FeatureFolder
# from compressai.datasets import ImageFolder
from compressai.losses import RateDistortionLoss
from compressai.optimizers import net_aux_optimizer
from compressai.zoo import image_models
import gc
# zlt
import math
import torch.nn.functional as F
from compressai.datasets import MultiSourceFeatureFolder
from typing import List, Tuple, Dict
from p2b_transform import feat_inverse_transform_torch, p2b_inverse_torch
from compressai.utils.eval_model.__main__ import load_all_layer_stats, load_all_p2b_stats, load_all_zscore_stats, load_all_quantization_mapping
# zlt
import os
import json
import warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
# === add: backbone manager for distillation ======================
import clip
warnings.filterwarnings("ignore", message="xFormers is available")

class BackboneManager:
    """
    Forward tokens from current ViT block output -> target block output (e.g., blk23).
    Model params are frozen; gradients flow through the blocks into rec_feat.
    """
    def __init__(self, device, dinov2_weights_root=None):
        self.device = device
        self.models = {}
        self.configs = {}
        if dinov2_weights_root is None:
            _sd = os.path.dirname(os.path.abspath(__file__))
            _pr = os.path.normpath(os.path.join(_sd, "..", "..", ".."))
            dinov2_weights_root = os.path.join(_pr, "pretrained")

        # Load DINOv2
        _sd = os.path.dirname(os.path.abspath(__file__))
        _pr = os.path.normpath(os.path.join(_sd, "..", "..", ".."))
        DINOV2_SOURCE_DIR = os.path.join(_pr, "backbone", "dinov2")
        if os.path.exists(DINOV2_SOURCE_DIR):
            sys.path.append(DINOV2_SOURCE_DIR)
            try:
                from dinov2.hub.classifiers import dinov2_vitl14_lc
                print("Loading DINOv2 ViT-L/14...")
                weights = []
                if dinov2_weights_root:
                    p1 = os.path.join(dinov2_weights_root, "dinov2_vitl14_pretrain.pth")
                    p2 = os.path.join(dinov2_weights_root, "dinov2_vitl14_linear_head.pth")
                    if os.path.exists(p1) and os.path.exists(p2):
                        weights = [p1, p2]

                if weights:
                    clf = dinov2_vitl14_lc(layers=1, pretrained=True, weights=weights)
                else:
                    print("Warning: DINOv2 weights not found, attempting default load...")
                    clf = dinov2_vitl14_lc(layers=1, pretrained=True)

                clf.to(device).eval()
                dino = getattr(clf, "backbone", clf)
                for p in dino.parameters():
                    p.requires_grad_(False)

                self.models["dinov2_vitl14"] = dino
                self.configs["dinov2_vitl14"] = {"num_blocks": 24}
            except Exception as e:
                print(f"Failed to load DINOv2: {e}")
        else:
            print(f"DINOv2 source not found at {DINOV2_SOURCE_DIR}")

        # Load CLIP
        if clip is not None:
            print("Loading CLIP ViT-L/14...")
            try:
                model, _ = clip.load("ViT-L/14", device=device)
                model = model.float().eval()
                for p in model.parameters():
                    p.requires_grad_(False)

                self.models["clip_vitl14"] = model
                self.configs["clip_vitl14"] = {"num_blocks": 24}
            except Exception as e:
                print(f"Failed to load CLIP: {e}")

    def run_to_layer(self, tokens, model_name: str, start_layer_idx: int, target_layer_idx: int):
        """
        tokens: [B, 257, C] = output tokens of block start_layer_idx
        Return: [B, 257, C] = output tokens of block target_layer_idx
        """
        if model_name not in self.models:
            return None
        if target_layer_idx <= start_layer_idx:
            return None

        model = self.models[model_name]
        total_blocks = self.configs[model_name]["num_blocks"]
        if target_layer_idx >= total_blocks:
            return None

        if model_name == "clip_vitl14":
            x = tokens.permute(1, 0, 2)  # [257, B, C]
            for idx in range(start_layer_idx + 1, target_layer_idx + 1):
                x = model.visual.transformer.resblocks[idx](x)
            return x.permute(1, 0, 2)

        if model_name == "dinov2_vitl14":
            x = tokens
            for idx in range(start_layer_idx + 1, target_layer_idx + 1):
                x = model.blocks[idx](x)
            return x

        return None
# === end add ======================================================

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""
    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": args.aux_learning_rate},
    }
    optimizer = net_aux_optimizer(net, conf)
    return optimizer["net"], optimizer["aux"]


def train_one_epoch(
        model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device

    for i, batch in enumerate(train_dataloader):
        d, orig_feat, backbones, layers = batch  # d: [B, 1, H, W] after ToTensor
        d = d.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 1000 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i * len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE: {out_criterion["mse_loss"].item():.7f} |'
                f'\tBpp: {out_criterion["bpp_loss"].item():.4f} |'
                f"\tAux: {aux_loss.item():.2f}"
            )


def test_epoch(epoch, test_dataloader, model, criterion):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            d, orig_feat, backbones, layers = batch  # d: [B, 1, H, W] after ToTensor
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE: {mse_loss.avg:.7f} |"
        f"\tBpp: {bpp_loss.avg:.4f} |"
        f"\tAux: {aux_loss.avg:.2f}"
    )

    return loss.avg, float(mse_loss.avg), float(bpp_loss.avg)

# # zlt version 6
# def train_one_epoch(
#         model, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm,
#         p2b_cache, lmbda: float, model_type: str, n_split: int = 64,
#         feat_transform: str = "p2b", zscore_cache=None,
#         backbone_mgr: "BackboneManager" = None,  # <-- add
#         distill_target_idx: int = 23,            # <-- add (blk23)
#         layer_stats_cache=None, bit_depth: int = 1
# ):
#     model.train()
#     device = next(model.parameters()).device
#
#     mse_meter = AverageMeter()
#     distill_meter = AverageMeter()
#     bpp_meter = AverageMeter()
#     loss_meter = AverageMeter()
#
#     for i, batch in enumerate(train_dataloader):
#         # batch: 来自 MultiSourceFeatureFolder.__getitem__
#         x_packed, orig_feat, backbones, layers = batch  # x_packed: [B, 1, H, W] after ToTensor
#
#         x_packed = x_packed.to(device)  # [B, 1, H, W]
#         orig_feat = orig_feat.to(device)  # [B, 257, C]
#
#         optimizer.zero_grad()
#         aux_optimizer.zero_grad()
#
#         # 1) codec 前向：输入仍是 P2B 后的 packed 特征
#         out_net = model(x_packed)
#         x_hat_packed = out_net["x_hat"]  # [B, 1, H, W]
#
#         # 2) 反 packing + P2B 逆变换，恢复到原空间 [B, 257, C]
#         #    先 unpack 到特征平面
#         feat_hat_packed = feat_inverse_transform_torch(x_hat_packed, n_split=n_split)  # [B, 257, C']
#         B, T, C = feat_hat_packed.shape
#
#         if feat_transform == "p2b":
#             rec_list = []
#             for b in range(B):
#                 backbone = backbones[b]
#                 layer = layers[b]
#                 p_np, qt_np = p2b_cache[(backbone, layer)]
#                 p_t = torch.from_numpy(p_np).to(device)
#                 qt_t = torch.from_numpy(qt_np).to(device)
#
#                 z_b = feat_hat_packed[b:b + 1]  # [1, T, C]  (p2b 域)
#                 x_hat_b = p2b_inverse_torch(z_b, p_t, qt_t)  # [1, T, C]  (真实 token 域)
#                 rec_list.append(x_hat_b.squeeze(0))
#
#             rec_feat = torch.stack(rec_list, dim=0)  # [B, T, C]
#             # 注意：orig_feat 现在来自 dataset，已经是真实 token 域，不要再动它
#
#         elif feat_transform == "zscore":
#             rec_list = []
#             for b in range(B):
#                 backbone = backbones[b]
#                 layer = layers[b]
#                 mean_t, std_t = zscore_cache[(backbone, layer)]  # [C], [C]
#                 mean_t = mean_t.to(device)
#                 std_t = std_t.to(device)
#
#                 x_hat_b = feat_hat_packed[b] * std_t.unsqueeze(0) + mean_t.unsqueeze(0)  # [T, C]
#                 rec_list.append(x_hat_b)
#
#             rec_feat = torch.stack(rec_list, dim=0)  # [B, T, C]
#             # orig_feat 同样不需要动
#
#         elif feat_transform == "trunc":
#             rec_list = []
#             for b in range(B):
#                 backbone = backbones[b]
#                 layer = layers[b]
#                 bounds = layer_stats_cache[backbone].get(layer, None)
#                 if bounds is None:
#                     raise KeyError(f"No stats for ({backbone}, {layer})")
#                 low = float(bounds["low"])
#                 high = float(bounds["high"])
#                 scale = ((2 ** bit_depth) - 1.0) / (high - low + 1e-12)
#
#                 q = feat_hat_packed[b]  # [T, C] in quant scale
#                 x_hat = q / scale + low  # dequant -> real token space
#                 x_hat = x_hat.clamp(low, high)  # 可选，和 trunc 语义一致
#                 rec_list.append(x_hat)
#
#             rec_feat = torch.stack(rec_list, dim=0)  # [B, T, C]
#
#         # === Distillation Loss: deep_gt no_grad, deep_hat grad ===
#         distill_loss = torch.tensor(0.0, device=device)
#         if backbone_mgr is not None:
#             distill_sum = torch.tensor(0.0, device=device)
#             distill_cnt = 0
#
#             unique_configs = set(zip(backbones, layers))
#             for (bb_name, layer_name) in unique_configs:
#                 try:
#                     l_idx = int(str(layer_name)[-2:])
#                 except Exception:
#                     continue
#
#                 if l_idx >= distill_target_idx:  # blk23: ignore
#                     distill_sum = distill_sum + F.mse_loss(rec_feat, orig_feat)
#                     continue
#
#                 idxs = [k for k, (bb, ll) in enumerate(zip(backbones, layers)) if bb == bb_name and ll == layer_name]
#                 if not idxs:
#                     continue
#                 idx_t = torch.tensor(idxs, device=device)
#
#                 sub_rec = rec_feat.index_select(0, idx_t)  # [b, T, C]
#                 sub_ori = orig_feat.index_select(0, idx_t)  # [b, T, C]
#
#                 # 1) GT: no grad
#                 with torch.no_grad():
#                     deep_gt = backbone_mgr.run_to_layer(sub_ori, bb_name, l_idx, distill_target_idx)
#                 if deep_gt is None:
#                     continue
#                 deep_gt = deep_gt.detach()
#
#                 # 2) Pred: keep graph (grad flows to sub_rec -> rec_feat -> codec)
#                 deep_hat = backbone_mgr.run_to_layer(sub_rec, bb_name, l_idx, distill_target_idx)
#                 if deep_hat is None:
#                     continue
#
#                 distill_sum = distill_sum + F.mse_loss(deep_hat, deep_gt)
#                 distill_cnt += deep_gt.numel()
#
#             if distill_cnt > 0:
#                 distill_loss = (255 ** 2) * (distill_sum / distill_cnt)
#
#         # # 3) 原空间 MSE
#         mse_loss = 255 ** 2 * F.mse_loss(rec_feat, orig_feat)
#
#         # 4) BPP 计算沿用 compressai 的公式
#         N, _, H, W = x_packed.size()
#         num_pixels = N * H * W
#         bpp_loss = 0.0
#         for _, likelihoods in out_net["likelihoods"].items():
#             bpp_loss += torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
#
#         loss = lmbda * (mse_loss + distill_loss) + bpp_loss
#         # loss = lmbda * distill_loss + bpp_loss
#
#         loss.backward()
#         if clip_max_norm > 0:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
#         optimizer.step()
#
#         aux_loss = model.aux_loss()
#         aux_loss.backward()
#         aux_optimizer.step()
#
#         mse_meter.update(mse_loss.item(), N)
#         distill_meter.update(distill_loss.item(), N)
#         bpp_meter.update(bpp_loss.item(), N)
#         loss_meter.update(loss.item(), N)
#
#         if i % 500 == 0:
#             print(
#                 f"Train epoch {epoch}: ["
#                 f"{i * N}/{len(train_dataloader.dataset)}"
#                 f" ({100. * i / len(train_dataloader):.0f}%)]"
#                 f"\tLoss: {loss_meter.val:.3f} |"
#                 f"\tMSE loss: {mse_meter.val:.3f} |"
#                 f"\tDstl loss: {distill_meter.val:.3f} |"
#                 f"\tBpp loss: {bpp_meter.val:.4f} |"
#                 f"\tAux loss: {aux_loss.item():.2f}"
#             )
#
#
# def test_epoch(model, test_dataloader, epoch, p2b_cache, lmbda: float,
#                model_type: str, n_split: int = 64,
#                feat_transform: str = "p2b", zscore_cache=None,
#                backbone_mgr=None, distill_target_idx: int = 23,
#                layer_stats_cache=None, bit_depth: int = 1):  # <-- add
#
#     model.eval()
#     device = next(model.parameters()).device
#
#     mse_meter = AverageMeter()
#     distill_meter = AverageMeter()
#     bpp_meter = AverageMeter()
#     loss_meter = AverageMeter()
#
#     with torch.no_grad():
#         for batch in test_dataloader:
#             x_packed, orig_feat, backbones, layers = batch
#             x_packed = x_packed.to(device)
#             orig_feat = orig_feat.to(device)
#
#             out_net = model(x_packed)
#             x_hat_packed = out_net["x_hat"]
#
#             feat_hat_packed = feat_inverse_transform_torch(x_hat_packed, n_split=n_split)  # [B, 257, C']
#             B, T, C = feat_hat_packed.shape
#
#             if feat_transform == "p2b":
#                 rec_list = []
#                 for b in range(B):
#                     backbone = backbones[b]
#                     layer = layers[b]
#                     p_np, qt_np = p2b_cache[(backbone, layer)]
#                     p_t = torch.from_numpy(p_np).to(device)
#                     qt_t = torch.from_numpy(qt_np).to(device)
#
#                     z_b = feat_hat_packed[b:b + 1]  # [1, T, C]  (p2b 域)
#                     x_hat_b = p2b_inverse_torch(z_b, p_t, qt_t)  # [1, T, C]  (真实 token 域)
#                     rec_list.append(x_hat_b.squeeze(0))
#
#                 rec_feat = torch.stack(rec_list, dim=0)  # [B, T, C]
#                 # 注意：orig_feat 现在来自 dataset，已经是真实 token 域，不要再动它
#
#             elif feat_transform == "zscore":
#                 rec_list = []
#                 for b in range(B):
#                     backbone = backbones[b]
#                     layer = layers[b]
#                     mean_t, std_t = zscore_cache[(backbone, layer)]  # [C], [C]
#                     mean_t = mean_t.to(device)
#                     std_t = std_t.to(device)
#
#                     x_hat_b = feat_hat_packed[b] * std_t.unsqueeze(0) + mean_t.unsqueeze(0)  # [T, C]
#                     rec_list.append(x_hat_b)
#
#                 rec_feat = torch.stack(rec_list, dim=0)  # [B, T, C]
#                 # orig_feat 同样不需要动
#
#             else:  # "trunc"（你说的 none 分支）
#                 rec_list = []
#                 for b in range(B):
#                     backbone = backbones[b]
#                     layer = layers[b]
#                     bounds = layer_stats_cache[backbone].get(layer, None)
#                     if bounds is None:
#                         raise KeyError(f"No stats for ({backbone}, {layer})")
#                     low = float(bounds["low"])
#                     high = float(bounds["high"])
#                     scale = ((2 ** bit_depth) - 1.0) / (high - low + 1e-12)
#
#                     q = feat_hat_packed[b]  # [T, C] in quant scale
#                     x_hat = q / scale + low  # dequant -> real token space
#                     x_hat = x_hat.clamp(low, high)  # 可选，和 trunc 语义一致
#                     rec_list.append(x_hat)
#
#                 rec_feat = torch.stack(rec_list, dim=0)  # [B, T, C]
#
#             # === Distillation Loss: push both rec_feat & orig_feat to blk23 ===
#             distill_loss = torch.tensor(0.0, device=device)
#             if backbone_mgr is not None:
#                 distill_sum = 0.0
#                 distill_cnt = 0
#
#                 unique_configs = set(zip(backbones, layers))
#                 for (bb_name, layer_name) in unique_configs:
#                     try:
#                         l_idx = int(str(layer_name)[-2:])  # 'blk05' -> 5
#                     except Exception:
#                         continue
#
#                     # blk23: ignore this loss
#                     if l_idx >= distill_target_idx:
#                         distill_sum = distill_sum + F.mse_loss(rec_feat, orig_feat)
#                         continue
#
#                     idxs = [k for k, (bb, ll) in enumerate(zip(backbones, layers)) if bb == bb_name and ll == layer_name]
#                     if not idxs:
#                         continue
#                     idx_t = torch.tensor(idxs, device=device)
#
#                     sub_rec = rec_feat.index_select(0, idx_t)
#                     sub_ori = orig_feat.index_select(0, idx_t)
#
#                     deep_gt = backbone_mgr.run_to_layer(sub_ori, bb_name, l_idx, distill_target_idx)
#                     deep_hat = backbone_mgr.run_to_layer(sub_rec, bb_name, l_idx, distill_target_idx)
#
#                     if deep_gt is None or deep_hat is None:
#                         continue
#
#                     distill_sum += F.mse_loss(deep_hat, deep_gt)
#                     distill_cnt += deep_gt.numel()
#
#                 if distill_cnt > 0:
#                     distill_loss = (255 ** 2) * (distill_sum / distill_cnt)
#
#             mse_loss = 255 ** 2 * F.mse_loss(rec_feat, orig_feat)
#
#             N, _, H, W = x_packed.size()
#             num_pixels = N * H * W
#             bpp_loss = 0.0
#             for _, likelihoods in out_net["likelihoods"].items():
#                 bpp_loss += torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
#
#             loss = lmbda * (mse_loss + distill_loss) + bpp_loss
#             # loss = lmbda * distill_loss + bpp_loss
#
#             mse_meter.update(mse_loss.item(), N)
#             distill_meter.update(distill_loss.item(), N)
#             bpp_meter.update(bpp_loss.item(), N)
#             loss_meter.update(loss.item(), N)
#
#     print(
#         f"Test epoch {epoch}: "
#         f"Loss: {loss_meter.avg:.3f} | "
#         f"MSE loss: {mse_meter.avg:.3f} | "
#         f"Distill loss: {distill_meter.avg:.3f} | "
#         f"Bpp loss: {bpp_meter.avg:.4f}"
#     )
#
#     return {
#         "loss": loss_meter.avg,
#         "mse_loss": mse_meter.avg,
#         "distill_loss": distill_meter.avg,
#         "bpp_loss": bpp_meter.avg,
#     }

# gcs
def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)

    if is_best:
        # gcs
        best_checkpoint_name = f"{filename[:-8]}_best.pth.tar"
        # print(best_checkpoint_name)
        shutil.copyfile(filename, best_checkpoint_name)

def plot_test_curves(epoch_list, mse_list, bpp_list, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax1 = plt.subplots()
    l1 = ax1.plot(epoch_list, mse_list, color="tab:blue", label="mse")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("test_mse", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.grid(True)

    ax2 = ax1.twinx()
    l2 = ax2.plot(epoch_list, bpp_list, color="tab:orange", label="bpp")
    ax2.set_ylabel("test_bpp", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    # 合并图例
    lines = l1 + l2
    labels = [ln.get_label() for ln in lines]
    ax1.legend(lines, labels, loc="best")

    plt.title("Test MSE & BPP")
    fig.tight_layout()

    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-m",
        "--model",
        default="bmshj2018-factorized",
        choices=image_models.keys(),
        help="Model architecture (default: %(default)s)",
    )
    # gcs, model_type="sd3", task="tti", trun_flag=False, trun_low=-20, trun_high=20, quant_type="uniform", qsamples=0, bit_depth=1
    parser.add_argument(
        "-model_type",
        "--model_type",
        type=str,
        default="sd3",
        help="Please input the model_type.",
    )
    parser.add_argument(
        "-task",
        "--task",
        type=str,
        default="tti",
        help="Please input the task.",
    )
    parser.add_argument(
        "-trun_flag",
        "--trun_flag",
        type=bool,
        default=True,
        help="Please input the trun_flag.",
    )
    parser.add_argument(
        "-trun_low",
        "--trun_low",
        type=float,
        default=-5,
        help="Please input the truncated lower value.",
    )
    parser.add_argument(
        "-trun_high",
        "--trun_high",
        type=float,
        default=5,
        help="Please input the truncated upper value.",
    )
    parser.add_argument(
        "-quant_type",
        "--quant_type",
        type=str,
        default="uniform",
        help="Please input the quant_type.",
    )
    parser.add_argument(
        "-qsamples",
        "--qsamples",
        type=int,
        default=0,
        help="Please input the qsamples.",
    )
    parser.add_argument(
        "-bit_depth",
        "--bit_depth",
        type=int,
        default=1,
        help="Please input the bit_depth.",
    )
    parser.add_argument(
        "-mp",
        "--savepath",
        type=str,
        help="Path to save trained models.",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, required=True, help="Training dataset"
    )
    parser.add_argument(
        "-e",
        "--epochs",
        default=100,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--learning-rate",
        "-lr",
        "--lr",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=8,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument(
        "--lambda",
        dest="lmbda",
        type=float,
        default=1e-2,
        help="Bit-rate distortion parameter (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size (default: %(default)s)"
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=64,
        help="Test batch size (default: %(default)s)",
    )
    parser.add_argument(
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(512, 512),  # (hgt, wdt)
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument("--seed", type=int, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    # zlt
    # === Add args (near other parser.add_argument) ==================
    parser.add_argument("-layers", "--layers",
                        type=str, nargs="*", default=None,
                        help="List of layers to aggregate, e.g., blk05 blk11 blk17 blk23")

    parser.add_argument("-backbone_name", "--backbone_name",
                        type=str, nargs="*", default=None,
                        help="The subfolder name under split/, e.g., dinov2_vitl14")
    parser.add_argument("--layer_stats_root", type=str, default=None,
                        help="Path to per-layer global min/max (json or npz)")
    parser.add_argument(
        "--train_split", type=str, required=True, help="Training dataset split"
    )
    parser.add_argument(
        "--test_split", type=str, required=True, help="Testing dataset split"
    )
    parser.add_argument(
        "--p2b_stats_root", type=str, default=None,
        help="p2b stat npz root"
    )
    # version5
    parser.add_argument(
        "--feat_transform",
        type=str,
        default="p2b",
        help="Feature pre-transform type: trunc | p2b | zscore",
    )
    parser.add_argument(
        "--zscore_stats_root",
        type=str,
        default=None,
        help="Root directory of zscore stats (npz files)",
    )
    # DT-UFC
    parser.add_argument(
        "--quantization_mapping_root",
        type=str,
        default=None,
        help="Root directory of quantization mapping (json files)",
    )
    parser.add_argument(
        "--save_curve_path",
        type=str,
        help="Path to save training curves.",
    )
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    # gcs
    train_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    # gcs
    # train_dataset = FeatureFolder(args.dataset, split="train", transform=train_transforms, model_type=args.model_type, task=args.task, trun_flag=args.trun_flag, trun_low=args.trun_low, trun_high=args.trun_high, quant_type=args.quant_type, qsamples=args.qsamples, bit_depth=args.bit_depth, patch_size=args.patch_size)
    # test_dataset = FeatureFolder(args.dataset, split="test", transform=test_transforms, model_type=args.model_type, task=args.task, trun_flag=args.trun_flag, trun_low=args.trun_low, trun_high=args.trun_high, quant_type=args.quant_type, qsamples=args.qsamples, bit_depth=args.bit_depth, patch_size=args.patch_size)
    # zlt
    if args.layers and len(args.layers) > 0:
        # multi-layer
        train_dataset = MultiSourceFeatureFolder(
            root=args.dataset,
            backbone_name=args.backbone_name,
            layers=args.layers,
            split=args.train_split,
            transform=train_transforms,
            model_type=args.model_type,
            task=args.task,
            trun_flag=args.trun_flag,
            trun_low=args.trun_low,
            trun_high=args.trun_high,
            quant_type=args.quant_type,
            qsamples=args.qsamples,
            bit_depth=args.bit_depth,
            patch_size=args.patch_size,
            layer_stats_root=args.layer_stats_root,
            feat_transform=args.feat_transform,  # version5
            zscore_stats_root=args.zscore_stats_root,  # version5
            quantization_mapping_root=args.quantization_mapping_root,
        )
        test_dataset = MultiSourceFeatureFolder(
            root=args.dataset,
            backbone_name=args.backbone_name,
            layers=args.layers,
            split=args.test_split,
            transform=test_transforms,
            model_type=args.model_type,
            task=args.task,
            trun_flag=args.trun_flag,
            trun_low=args.trun_low,
            trun_high=args.trun_high,
            quant_type=args.quant_type,
            qsamples=args.qsamples,
            bit_depth=args.bit_depth,
            patch_size=args.patch_size,
            layer_stats_root=args.layer_stats_root,
            feat_transform=args.feat_transform,  # version5
            zscore_stats_root=args.zscore_stats_root,  # version5
            quantization_mapping_root=args.quantization_mapping_root,
        )
    else:
        train_dataset = FeatureFolder(args.dataset, split="train", transform=train_transforms,
                                      model_type=args.model_type, task=args.task, trun_flag=args.trun_flag,
                                      trun_low=args.trun_low, trun_high=args.trun_high, quant_type=args.quant_type,
                                      qsamples=args.qsamples, bit_depth=args.bit_depth, patch_size=args.patch_size)
        test_dataset = FeatureFolder(args.dataset, split="test", transform=test_transforms, model_type=args.model_type,
                                     task=args.task, trun_flag=args.trun_flag, trun_low=args.trun_low,
                                     trun_high=args.trun_high, quant_type=args.quant_type, qsamples=args.qsamples,
                                     bit_depth=args.bit_depth, patch_size=args.patch_size)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    net = image_models[args.model](quality=1)
    # backbone_mgr = BackboneManager(device=torch.device(device))  # version 6
    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    # === version5: 根据 feat_transform 预加载 P2B / Z-score 统计 ===
    # p2b_cache = None
    # zscore_cache = None
    # layer_stats_cache = None
    # if args.feat_transform == "trunc":
    #     layer_stats_cache = load_all_layer_stats(args.layer_stats_root, args.backbone_name)
    # elif args.feat_transform == "p2b":
    #     p2b_cache = load_all_p2b_stats(args.p2b_stats_root, args.backbone_name, args.layers)
    # elif args.feat_transform == "zscore":
    #     zscore_cache = load_all_zscore_stats(args.zscore_stats_root, args.backbone_name, args.layers, device)
    # else: # 'kmeans' / 'density' / 'blend' / 'ekmeans'
    #     quantization_cache = load_all_quantization_mapping(args.quantization_mapping_root, args.backbone_name, args.layers, args.feat_transform)
    #     layer_stats_cache = load_all_layer_stats(args.layer_stats_root, args.backbone_name)

    # lmbda = args.lmbda

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    epoch_history = []
    test_mse_history = []
    test_bpp_history = []

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        # train_one_epoch(
        #     net, train_dataloader, optimizer, aux_optimizer, epoch, args.clip_max_norm,
        #     p2b_cache=p2b_cache, lmbda=lmbda, model_type=args.model_type,
        #     feat_transform=args.feat_transform, zscore_cache=zscore_cache,
        #     backbone_mgr=backbone_mgr,          # <-- add
        #     distill_target_idx=23,              # <-- add
        #     layer_stats_cache=layer_stats_cache, bit_depth=args.bit_depth,  # <-- add
        # )
        #
        # loss_dict = test_epoch(
        #     net, test_dataloader, epoch,
        #     p2b_cache=p2b_cache, lmbda=lmbda, model_type=args.model_type,
        #     feat_transform=args.feat_transform, zscore_cache=zscore_cache,
        #     backbone_mgr=backbone_mgr, distill_target_idx=23,  # <-- add
        #     layer_stats_cache=layer_stats_cache, bit_depth=args.bit_depth,  # <-- add
        # )
        # loss = loss_dict["loss"]

        criterion = RateDistortionLoss(lmbda=args.lmbda)  # for original mse loss
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )
        loss, mse_avg, bpp_avg = test_epoch(epoch, test_dataloader, net, criterion)
        epoch_history.append(epoch)
        test_mse_history.append(mse_avg)
        test_bpp_history.append(bpp_avg)

        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        # gcs
        gc.collect()  # Run Python garbage collection
        torch.cuda.empty_cache()  # Then clear cached memory on GPU

        if epoch % 100 == 0 or epoch >= args.epochs - 1:
            if args.save:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best,
                    # gcs
                    args.savepath,
                )
        if epoch % 10 == 0 or epoch >= args.epochs - 1:
            # ✅ 保存曲线图到 checkpoint 同一路径
            plot_test_curves(epoch_history, test_mse_history, test_bpp_history, args.save_curve_path)

if __name__ == "__main__":
    main(sys.argv[1:])

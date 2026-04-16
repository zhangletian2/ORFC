# Copyright (c) 2021-2024, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
from compressai.utils.eval_model.__main__ import load_all_p2b_stats, load_all_zscore_stats


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


# zlt version 5
def train_one_epoch(
        model, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm,
        p2b_cache, lmbda: float, model_type: str, n_split: int = 64,
        feat_transform: str = "p2b", zscore_cache=None,
):
    model.train()
    device = next(model.parameters()).device

    mse_meter = AverageMeter()
    bpp_meter = AverageMeter()
    loss_meter = AverageMeter()

    for i, batch in enumerate(train_dataloader):
        # batch: 来自 MultiSourceFeatureFolder.__getitem__
        x_packed, orig_feat, backbones, layers = batch  # x_packed: [B, 1, H, W] after ToTensor

        x_packed = x_packed.to(device)  # [B, 1, H, W]
        orig_feat = orig_feat.to(device)  # [B, 257, C]

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        # 1) codec 前向：输入仍是 P2B 后的 packed 特征
        out_net = model(x_packed)
        x_hat_packed = out_net["x_hat"]  # [B, 1, H, W]

        # 2) 反 packing + P2B 逆变换，恢复到原空间 [B, 257, C]
        #    先 unpack 到特征平面
        feat_hat_packed = feat_inverse_transform_torch(x_hat_packed, n_split=n_split)  # [B, 257, C']
        B, T, C = feat_hat_packed.shape

        # === version5: 根据 feat_transform 选择逆变换 ===
        if feat_transform == "p2b":
            rec_list = []
            ori_list = []  # 12.2 debug
            for b in range(B):
                backbone = backbones[b]
                layer = layers[b]
                p_np, qt_np = p2b_cache[(backbone, layer)]
                p_t = torch.from_numpy(p_np).to(device)
                qt_t = torch.from_numpy(qt_np).to(device)

                z_b = feat_hat_packed[b:b + 1]  # [1, T, C]
                x_hat_b = p2b_inverse_torch(z_b, p_t, qt_t)  # [1, T, C]
                rec_list.append(x_hat_b.squeeze(0))  # [T, C]

                z_ori_b = orig_feat[b:b+1]  # 12.2 debug
                x_ori_hat_b = p2b_inverse_torch(z_ori_b, p_t, qt_t)  # 12.2 debug
                ori_list.append(x_ori_hat_b.squeeze(0))  # 12.2 debug

            rec_feat = torch.stack(rec_list, dim=0)  # [B, T, C]
            orig_feat = torch.stack(ori_list, dim=0)  # 12.2 debug

        elif feat_transform == "zscore":
            if zscore_cache is None:
                raise RuntimeError("feat_transform='zscore' 但 zscore_cache 为空")
            rec_list = []
            ori_list = []  # 12.2 debug
            for b in range(B):
                backbone = backbones[b]
                layer = layers[b]
                mean_t, std_t = zscore_cache[(backbone, layer)]  # [C], [C] 已在 device 上
                mean_t = mean_t.to(device)
                std_t = std_t.to(device)
                x_hat_b = feat_hat_packed[b] * std_t.unsqueeze(0) + mean_t.unsqueeze(0)  # [T,C]
                rec_list.append(x_hat_b)

                x_ori_hat_b = orig_feat[b] * std_t.unsqueeze(0) + mean_t.unsqueeze(0)  # 12.2 debug
                ori_list.append(x_ori_hat_b)  # 12.2 debug

            rec_feat = torch.stack(rec_list, dim=0)  # [B,T,C]
            orig_feat = torch.stack(ori_list, dim=0)  # 12.2 debug

        else:  # 'none'
            # codec 直接在原空间工作，feat_hat_packed 就是重建特征
            rec_feat = feat_hat_packed

        # 3) 原空间 MSE
        mse_loss = 255 ** 2 * F.mse_loss(rec_feat, orig_feat)

        # 4) BPP 计算沿用 compressai 的公式
        N, _, H, W = x_packed.size()
        num_pixels = N * H * W
        bpp_loss = 0.0
        for _, likelihoods in out_net["likelihoods"].items():
            bpp_loss += torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)

        loss = lmbda * mse_loss + bpp_loss

        loss.backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        mse_meter.update(mse_loss.item(), N)
        bpp_meter.update(bpp_loss.item(), N)
        loss_meter.update(loss.item(), N)

        if i % 500 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i * N}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f"\tLoss: {loss_meter.val:.3f} |"
                f"\tMSE loss: {mse_meter.val:.7f} |"
                f"\tBpp loss: {bpp_meter.val:.4f} |"
                f"\tAux loss: {aux_loss.item():.2f}"
            )


def test_epoch(model, test_dataloader, epoch, p2b_cache, lmbda: float,
               model_type: str, n_split: int = 64,
               feat_transform: str = "p2b", zscore_cache=None):
    model.eval()
    device = next(model.parameters()).device

    mse_meter = AverageMeter()
    bpp_meter = AverageMeter()
    loss_meter = AverageMeter()

    with torch.no_grad():
        for batch in test_dataloader:
            x_packed, orig_feat, backbones, layers = batch
            x_packed = x_packed.to(device)
            orig_feat = orig_feat.to(device)

            out_net = model(x_packed)
            x_hat_packed = out_net["x_hat"]

            feat_hat_packed = feat_inverse_transform_torch(x_hat_packed, n_split=n_split)  # [B, 257, C']
            B, T, C = feat_hat_packed.shape

            # === version5: 根据 feat_transform 选择逆变换 ===
            if feat_transform == "p2b":
                # rec_list = []
                # for b in range(B):
                #     backbone = backbones[b]
                #     layer = layers[b]
                #     p_np, qt_np = p2b_cache[(backbone, layer)]
                #     p_t = torch.from_numpy(p_np).to(device)
                #     qt_t = torch.from_numpy(qt_np).to(device)
                #
                #     z_b = feat_hat_packed[b:b + 1]  # [1, T, C]
                #     x_hat_b = p2b_inverse_torch(z_b, p_t, qt_t)  # [1, T, C]
                #     rec_list.append(x_hat_b.squeeze(0))  # [T, C]
                #
                # rec_feat = torch.stack(rec_list, dim=0)  # [B, T, C]
                rec_feat = feat_hat_packed  # 12.1 debug

            elif feat_transform == "zscore":
                # if zscore_cache is None:
                #     raise RuntimeError("feat_transform='zscore' 但 zscore_cache 为空")
                # rec_list = []
                # for b in range(B):
                #     backbone = backbones[b]
                #     layer = layers[b]
                #     mean_t, std_t = zscore_cache[(backbone, layer)]  # [C], [C] 已在 device 上
                #     mean_t = mean_t.to(device)
                #     std_t = std_t.to(device)
                #     x_hat_b = feat_hat_packed[b] * std_t.unsqueeze(0) + mean_t.unsqueeze(0)  # [T,C]
                #     rec_list.append(x_hat_b)
                # rec_feat = torch.stack(rec_list, dim=0)  # [B,T,C]
                rec_feat = feat_hat_packed  # 12.1 debug


            else:  # 'none'
                # codec 直接在原空间工作，feat_hat_packed 就是重建特征
                rec_feat = feat_hat_packed

            mse_loss = 255 ** 2 * F.mse_loss(rec_feat, orig_feat)

            N, _, H, W = x_packed.size()
            num_pixels = N * H * W
            bpp_loss = 0.0
            for _, likelihoods in out_net["likelihoods"].items():
                bpp_loss += torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)

            loss = lmbda * mse_loss + bpp_loss

            mse_meter.update(mse_loss.item(), N)
            bpp_meter.update(bpp_loss.item(), N)
            loss_meter.update(loss.item(), N)

    print(
        f"Test epoch {epoch}: "
        f"Loss: {loss_meter.avg:.3f} | "
        f"MSE loss: {mse_meter.avg:.7f} | "
        f"Bpp loss: {bpp_meter.avg:.4f}"
    )

    return {
        "loss": loss_meter.avg,
        "mse_loss": mse_meter.avg,
        "bpp_loss": bpp_meter.avg,
    }


# def train_one_epoch(
#         model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
# ):
#     model.train()
#     device = next(model.parameters()).device
#
#     for i, d in enumerate(train_dataloader):
#         d = d.to(device)
#
#         optimizer.zero_grad()
#         aux_optimizer.zero_grad()
#
#         out_net = model(d)
#
#         out_criterion = criterion(out_net, d)
#         out_criterion["loss"].backward()
#         if clip_max_norm > 0:
#             torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
#         optimizer.step()
#
#         aux_loss = model.aux_loss()
#         aux_loss.backward()
#         aux_optimizer.step()
#
#         if i % 500 == 0:
#             print(
#                 f"Train epoch {epoch}: ["
#                 f"{i * len(d)}/{len(train_dataloader.dataset)}"
#                 f" ({100. * i / len(train_dataloader):.0f}%)]"
#                 f'\tLoss: {out_criterion["loss"].item():.3f} |'
#                 f'\tMSE loss: {out_criterion["mse_loss"].item():.7f} |'
#                 f'\tBpp loss: {out_criterion["bpp_loss"].item():.4f} |'
#                 f"\tAux loss: {aux_loss.item():.2f}"
#             )
#
#
# def test_epoch(epoch, test_dataloader, model, criterion):
#     model.eval()
#     device = next(model.parameters()).device
#
#     loss = AverageMeter()
#     bpp_loss = AverageMeter()
#     mse_loss = AverageMeter()
#     aux_loss = AverageMeter()
#
#     with torch.no_grad():
#         for d in test_dataloader:
#             d = d.to(device)
#             out_net = model(d)
#             out_criterion = criterion(out_net, d)
#
#             aux_loss.update(model.aux_loss())
#             bpp_loss.update(out_criterion["bpp_loss"])
#             loss.update(out_criterion["loss"])
#             mse_loss.update(out_criterion["mse_loss"])
#
#     print(
#         f"Test epoch {epoch}: Average losses:"
#         f"\tLoss: {loss.avg:.3f} |"
#         f"\tMSE loss: {mse_loss.avg:.7f} |"
#         f"\tBpp loss: {bpp_loss.avg:.4f} |"
#         f"\tAux loss: {aux_loss.avg:.2f}\n"
#     )
#
#     return loss.avg


# gcs
def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

    if is_best:
        # gcs
        best_checkpoint_name = f"{filename[:-8]}_best.pth.tar"
        # print(best_checkpoint_name)
        shutil.copyfile(filename, best_checkpoint_name)


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
        default=4,
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
    parser.add_argument("--layer_stats", type=str, nargs="*", default=None,
                        help="Path to per-layer global min/max (json or npz)")
    parser.add_argument(
        "--train_split", type=str, required=True, help="Training dataset split"
    )
    parser.add_argument(
        "--test_split", type=str, required=True, help="Testing dataset split"
    )
    parser.add_argument(
        "--p2b_stats", type=str, default=None,
        help="p2b stat npz root"
    )
    # version5
    parser.add_argument(
        "--feat_transform",
        type=str,
        default="p2b",
        choices=["trunc", "p2b", "zscore"],
        help="Feature pre-transform type: trunc | p2b | zscore",
    )
    parser.add_argument(
        "--zscore_stats",
        type=str,
        default=None,
        help="Root directory of zscore stats (npz files)",
    )

    # ================================================================

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
            layer_stats_path=args.layer_stats,
            feat_transform=args.feat_transform,  # version5
            zscore_stats_root=args.zscore_stats,  # version5
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
            layer_stats_path=args.layer_stats,
            feat_transform=args.feat_transform,  # version5
            zscore_stats_root=args.zscore_stats,  # version5
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
    net = net.to(device)

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    # === version5: 根据 feat_transform 预加载 P2B / Z-score 统计 ===
    p2b_cache = None
    zscore_cache = None
    if args.feat_transform == "p2b":
        p2b_cache = load_all_p2b_stats(
            stats_root=args.p2b_stats,
            model_type=args.backbone_name,
            layers=args.layers,
        )
    elif args.feat_transform == "zscore":
        if args.zscore_stats is None:
            raise ValueError("feat_transform='zscore' 但未提供 --zscore_stats")
        zscore_cache = load_all_zscore_stats(
            stats_root=args.zscore_stats,
            model_types=args.backbone_name,
            layers=args.layers,
            device=device,
        )
    lmbda = args.lmbda

    # criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint:  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net, train_dataloader, optimizer, aux_optimizer, epoch, args.clip_max_norm,
            p2b_cache=p2b_cache, lmbda=lmbda, model_type=args.model_type,
            feat_transform=args.feat_transform, zscore_cache=zscore_cache,
        )
        loss_dict = test_epoch(
            net, test_dataloader, epoch,
            p2b_cache=p2b_cache, lmbda=lmbda, model_type=args.model_type,
            feat_transform=args.feat_transform, zscore_cache=zscore_cache,
        )

        loss = loss_dict["loss"]
        # train_one_epoch(
        #     net,
        #     criterion,
        #     train_dataloader,
        #     optimizer,
        #     aux_optimizer,
        #     epoch,
        #     args.clip_max_norm,
        # )
        # loss = test_epoch(epoch, test_dataloader, net, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        # gcs
        gc.collect()  # Run Python garbage collection
        torch.cuda.empty_cache()  # Then clear cached memory on GPU

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


if __name__ == "__main__":
    main(sys.argv[1:])

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
import json
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from torchvision import transforms

#gcs
from compressai.datasets import FeatureFolder
from compressai.datasets import SegFeatureFolder
# from compressai.datasets import ImageFolder
from compressai.losses import RateDistortionLoss
from compressai.ops import compute_padding
from compressai.optimizers import net_aux_optimizer
from compressai.zoo import image_models
from compressai.zoo.image import model_architectures
import gc
import numpy as np
import sys
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
sys.path.append(os.path.join(_PROJECT_ROOT, 'coding', 'vq'))
from backbone.wrapper import Dinov2Wrapper, ClipWrapper
import warnings
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

# ========================= 分割任务相关 =========================

VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
NUM_SEG_CLASSES = 21
IGNORE_INDEX = 255


class IOUMetric:
    """计算mIoU的累加器"""
    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, target):
        """pred, target: numpy arrays [H, W]"""
        mask = target != self.ignore_index
        pred = pred[mask].astype(np.int64)
        target = target[mask].astype(np.int64)
        indices = self.num_classes * target + pred
        cm = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += cm.reshape(self.num_classes, self.num_classes)

    def compute(self):
        """计算mIoU和每类IoU"""
        intersection = np.diag(self.confusion_matrix)
        union = (self.confusion_matrix.sum(axis=1) +
                 self.confusion_matrix.sum(axis=0) - intersection)
        iou = intersection / (union + 1e-10)
        valid = union > 0
        miou = iou[valid].mean()
        total = self.confusion_matrix.sum()
        correct = intersection.sum()
        acc = correct / (total + 1e-10)
        return miou, acc, iou


def _get_layer_idx(dinowrapper_or_layer_str):
    """从 layer 字符串或对象中提取层索引"""
    if isinstance(dinowrapper_or_layer_str, str):
        digits = "".join([c for c in dinowrapper_or_layer_str if c.isdigit()])
        return int(digits) if digits else 0
    return 0

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
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm, writer
):
    model.train()
    device = next(model.parameters()).device

    total_loss = 0.0
    total_bpp = 0.0
    total_mse = 0.0
    total_aux = 0.0
    num_batches = 0

    for i, d in enumerate(train_dataloader):
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

        total_loss += out_criterion["loss"].detach()
        total_bpp += out_criterion["bpp_loss"].detach()
        total_mse += out_criterion["mse_loss"].detach()
        total_aux += aux_loss.detach()
        num_batches += 1

    avg_loss = (total_loss / num_batches).item()
    avg_bpp = (total_bpp / num_batches).item()
    avg_mse = (total_mse / num_batches).item()
    avg_aux = (total_aux / num_batches).item()

    print(
        f"Train epoch {epoch}:"
        f"\tLoss: {avg_loss:.3f} |"
        f"\tMSE loss: {avg_mse:.4f} |"
        f"\tBpp loss: {avg_bpp:.4f} |"
        f"\tAux loss: {avg_aux:.2f}"
    )

    if writer is not None:
        writer.add_scalar("train/loss", avg_loss, epoch)
        writer.add_scalar("train/mse_loss", avg_mse, epoch)
        writer.add_scalar("train/bpp_loss", avg_bpp, epoch)
        writer.add_scalar("train/aux_loss", avg_aux, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)

def test_epoch(epoch, test_dataloader, model, dinowrapper, criterion, layer_idx, model_type, bit_depth, writer, run_cls_eval=True):
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in test_dataloader:
            d, label, norm_params, orig_shape, org_feat = batch[0], batch[1], batch[2], batch[3], batch[4]
            d = d.to(device)

            h, w = d.size(2), d.size(3)
            pad, unpad = compute_padding(h, w, min_div=2**6)
            d_padded = F.pad(d, pad, mode="constant", value=0)

            out_net = model(d_padded)
            x_hat_raw = out_net["x_hat"]
            x_hat = F.pad(x_hat_raw, unpad)

            num_points = d.size(0) * d.size(1) * d.size(2) * d.size(3)
            bpp = 0.0
            if num_points > 0:
                for key in out_net["likelihoods"]:
                    lh = out_net["likelihoods"][key]
                    lh = lh.clamp(min=1e-9)
                    bpp += -lh.log2().sum().item() / num_points

            batch_size = d.size(0)
            x_hat_np = x_hat.squeeze(1).cpu().numpy()

            if orig_shape is not None:
                orig_shape_np = orig_shape.cpu().numpy()
                x_hat_np = np.stack([
                    FeatureFolder.unpacking(x_hat_np[i], tuple(int(v) for v in orig_shape_np[i]), model_type)
                    for i in range(batch_size)
                ])
            norm_params_np = norm_params.cpu().numpy()
            trun_low = float(norm_params_np[0, 0])
            trun_high = float(norm_params_np[0, 1])
            x_hat_np = FeatureFolder.uniform_dequantization(x_hat_np, trun_low, trun_high, bit_depth)

            org_feat_np = org_feat.cpu().numpy() if torch.is_tensor(org_feat) else org_feat.numpy()
            x_rec = torch.from_numpy(org_feat_np).to(device)
            x_hat_rec = torch.from_numpy(x_hat_np).to(device)

            mse = torch.mean((x_hat_rec - x_rec) ** 2)
            dist = (255 ** 2) * mse
            rd_loss = criterion.lmbda * dist + bpp

            aux_loss.update(model.aux_loss().item())
            bpp_loss.update(bpp)
            loss.update(rd_loss.item())
            mse_loss.update(mse.item())

            if run_cls_eval:
                label = label.to(device)
                if x_hat_rec.dim() == 4:
                    feat_hat = x_hat_rec.squeeze(1)
                elif x_hat_rec.dim() == 3:
                    feat_hat = x_hat_rec
                else:
                    feat_hat = x_hat_rec.unsqueeze(0)
                if feat_hat.dim() == 3 and feat_hat.shape[1] != 257 and feat_hat.shape[2] == 257:
                    feat_hat = feat_hat.permute(0, 2, 1).contiguous()
                logits = dinowrapper.forward_from_tokens(feat_hat, layer_idx)
                pred = torch.max(logits, 1)[1]
                total_correct += (pred == label).sum().item()
                total_samples += label.size(0)

    acc_str = "N/A"
    if run_cls_eval and total_samples > 0:
        acc = total_correct / total_samples
        acc_str = f"{acc:.4f}"

    print(
        f"Test epoch {epoch}:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.4f} |"
        f"\tBPFP: {bpp_loss.avg:.4f} |"
        f"\tAux loss: {aux_loss.avg:.2f} |"
        f"\tAcc: {acc_str}\n"
    )

    if writer is not None:
        writer.add_scalar("test/loss", loss.avg, epoch)
        writer.add_scalar("test/mse_loss", mse_loss.avg, epoch)
        writer.add_scalar("test/bpfp", bpp_loss.avg, epoch)
        writer.add_scalar("test/aux_loss", aux_loss.avg, epoch)
        if run_cls_eval and total_samples > 0:
            writer.add_scalar("test/acc", total_correct / total_samples, epoch)

    return loss.avg


def _slide_inference_seg(feat_hat_3d, gt_np, crops, dinowrapper, layer_idx, seg_head, device):
    """
    单张图片的 slide inference 分割评估
    
    Args:
        feat_hat_3d: numpy [num_slides, 1+N, D] 解压后的原始特征
        gt_np: numpy [H_gt, W_gt] GT 分割图
        crops: numpy [num_slides, 4] 预计算的滑窗坐标 (y1, x1, y2, x2)
        dinowrapper: Dinov2Wrapper（含 backbone 用于继续前向）
        layer_idx: 起始 block 索引
        seg_head: 分割头
        device: 计算设备
    
    Returns:
        pred: numpy [H_gt, W_gt] 预测分割图
    """
    import math
    
    PATCH_SIZE = SegFeatureFolder.PATCH_SIZE  # 14
    CROP_SIZE = SegFeatureFolder.CROP_SIZE    # (512, 512)
    h_crop, w_crop = CROP_SIZE
    
    orig_h, orig_w = gt_np.shape[:2]
    # 复现 mmseg resize 后的图片尺寸（与特征提取时一致）
    h_img, w_img = SegFeatureFolder.compute_resized_shape(orig_h, orig_w)
    
    num_slides = feat_hat_3d.shape[0]
    assert num_slides == len(crops), f"slides 数不匹配: feat={num_slides} vs crops={len(crops)}"
    
    # 初始化融合矩阵（在 resize 后的图片空间上）
    preds = torch.zeros((1, NUM_SEG_CLASSES, h_img, w_img), device=device)
    count_mat = torch.zeros((1, 1, h_img, w_img), device=device)
    
    for s, (y1, x1, y2, x2) in enumerate(crops):
        # 取该 slide 的特征 [1+N, D]
        tokens = torch.from_numpy(feat_hat_3d[s]).float().to(device).unsqueeze(0)  # [1, 1+N, D]
        
        # 计算 CenterPadding 后的 feat 空间尺寸（与 wrapper.py slide_inference_decode 一致）
        actual_h = y2 - y1
        actual_w = x2 - x1
        padded_crop_h = math.ceil(actual_h / PATCH_SIZE) * PATCH_SIZE
        padded_crop_w = math.ceil(actual_w / PATCH_SIZE) * PATCH_SIZE
        feat_h = padded_crop_h // PATCH_SIZE
        feat_w = padded_crop_w // PATCH_SIZE
        
        # 通过 backbone 继续前向 + norm
        logits = dinowrapper.forward_from_tokens_seg(tokens, layer_idx, feat_h, feat_w)
        
        # 上采样到 crop 尺寸，裁剪到实际尺寸
        logits_up = F.interpolate(logits, size=(h_crop, w_crop), mode='bilinear', align_corners=False)
        logits_crop = logits_up[:, :, :actual_h, :actual_w]
        
        # 累加到全图
        preds[:, :, y1:y2, x1:x2] += logits_crop
        count_mat[:, :, y1:y2, x1:x2] += 1
    
    # 平均
    assert (count_mat == 0).sum() == 0, "count_mat 有零值"
    preds = preds / count_mat
    
    # Resize 预测回 GT 尺寸
    if (h_img, w_img) != (orig_h, orig_w):
        preds = F.interpolate(preds, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
    
    pred = preds.argmax(dim=1).squeeze(0).cpu().numpy()
    return pred


def test_epoch_seg(epoch, test_dataloader, model, dinowrapper, seg_head, layer_idx, criterion, model_type, bit_depth, writer, run_seg_eval=True):
    """分割任务的测试 epoch（slide inference 融合，batch_size=1）
    
    Args:
        run_seg_eval: 是否执行 slide inference 分割评估。
                      False 时仅计算 RD loss，跳过耗时的 DINOv2 前向，大幅加速。
    """
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    metric = IOUMetric(NUM_SEG_CLASSES, IGNORE_INDEX) if run_seg_eval else None

    with torch.no_grad():
        for d, gt, norm_params, orig_shape, crops, org_feat in test_dataloader:
            # batch_size=1: d [1,1,1+N,num_slides*D], gt [1,H,W], crops [1,num_slides,4]
            d = d.to(device)

            h, w = d.size(2), d.size(3)
            pad, unpad = compute_padding(h, w, min_div=2**6)
            d_padded = F.pad(d, pad, mode="constant", value=0)

            out_net = model(d_padded)
            x_hat = F.pad(out_net["x_hat"], unpad)

            # 计算 BPP
            num_points = d.size(2) * d.size(3)
            bpp = 0.0
            if num_points > 0:
                for key in out_net["likelihoods"]:
                    lh = out_net["likelihoods"][key]
                    lh = lh.clamp(min=1e-9)
                    bpp += -lh.log2().sum().item() / num_points

            # 反变换（squeeze 到单样本）
            x_hat_np = x_hat[0, 0].cpu().numpy()
            
            trun_low = float(norm_params[0, 0])
            trun_high = float(norm_params[0, 1])
            x_hat_np = FeatureFolder.uniform_dequantization(x_hat_np, trun_low, trun_high, bit_depth)

            # MSE / RD loss (against original pre-truncation feature)
            org_feat_np = org_feat[0].cpu().numpy() if torch.is_tensor(org_feat) else org_feat[0].numpy()
            x_rec = torch.from_numpy(org_feat_np).to(device)
            x_hat_rec = torch.from_numpy(x_hat_np).to(device)
            mse = torch.mean((x_hat_rec - x_rec) ** 2)
            dist = (255 ** 2) * mse
            rd_loss = criterion.lmbda * dist + bpp

            aux_loss.update(model.aux_loss().item())
            bpp_loss.update(bpp)
            loss.update(rd_loss.item())
            mse_loss.update(mse.item())

            # 分割评估：unpacking + slide inference（仅在 run_seg_eval 时执行）
            if run_seg_eval:
                gt_np = gt[0].cpu().numpy()
                orig_shape_i = tuple(int(v) for v in orig_shape[0].cpu().numpy())
                crops_np = crops[0].cpu().numpy()  # [num_slides, 4]
                
                feat_hat_3d = SegFeatureFolder.seg_unpacking(x_hat_np, orig_shape_i)
                pred = _slide_inference_seg(
                    feat_hat_3d, gt_np, crops_np, dinowrapper,
                    layer_idx, seg_head, device
                )
                metric.update(pred, gt_np)

    miou_str = "N/A"
    if run_seg_eval:
        miou, acc, class_iou = metric.compute()
        miou_str = f"{miou:.4f}"
    
    print(
        f"Test epoch {epoch}:"
        f"\tLoss: {loss.avg:.3f} |"
        f"\tMSE loss: {mse_loss.avg:.4f} |"
        f"\tBPFP: {bpp_loss.avg:.4f} |"
        f"\tAux loss: {aux_loss.avg:.2f} |"
        f"\tmIoU: {miou_str}\n"
    )

    if writer is not None:
        writer.add_scalar("test/loss", loss.avg, epoch)
        writer.add_scalar("test/mse_loss", mse_loss.avg, epoch)
        writer.add_scalar("test/bpfp", bpp_loss.avg, epoch)
        writer.add_scalar("test/aux_loss", aux_loss.avg, epoch)
        if run_seg_eval:
            writer.add_scalar("test/miou", miou, epoch)

    return loss.avg


#gcs
def get_best_checkpoint_path(filename):
    """根据 checkpoint 路径生成 best checkpoint 路径，更鲁棒的命名"""
    filepath = Path(filename)
    stem = filepath.stem  # 不含后缀的文件名
    # 移除可能存在的 .pth 后缀（处理 .pth.tar 的情况）
    if stem.endswith('.pth'):
        stem = stem[:-4]
    best_name = f"{stem}_best.pth.tar"
    return str(filepath.parent / best_name)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    # 确保父目录存在
    filepath = Path(filename)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save(state, filename)
    
    if is_best:
        best_checkpoint_path = get_best_checkpoint_path(filename)
        shutil.copyfile(filename, best_checkpoint_path)


@torch.no_grad()
def final_eval(model, test_dataloader, dinowrapper, layer_idx, model_type, bit_depth, savepath):
    """
    训练完成后，加载 best checkpoint 进行真实编码测试。
    使用 compress/decompress 计算真实码率、Acc、编解码时间。
    """
    model.eval()
    device = next(model.parameters()).device

    # 更新熵模型
    if hasattr(model, "update"):
        model.update(force=True)

    bpfp_meter = AverageMeter()
    mse_meter = AverageMeter()
    enc_time_meter = AverageMeter()
    dec_time_meter = AverageMeter()
    total_correct = 0
    total_samples = 0
    nan_count = 0

    print("\n" + "="*60)
    print("Final Evaluation with Real Entropy Coding")
    print("="*60)

    for batch_idx, batch in enumerate(test_dataloader):
        d, label, norm_params, orig_shape, org_feat = batch[0], batch[1], batch[2], batch[3], batch[4]
        d = d.to(device)
        batch_size = d.size(0)

        # 逐样本处理（compress/decompress 需要逐个处理）
        for i in range(batch_size):
            x = d[i : i + 1]
            h, w = x.size(2), x.size(3)
            pad, unpad = compute_padding(h, w, min_div=2**6)
            x_padded = F.pad(x, pad, mode="constant", value=0)

            # 真实编码
            start_enc = time.time()
            out_enc = model.compress(x_padded)
            enc_time = time.time() - start_enc

            # 真实解码
            start_dec = time.time()
            out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
            dec_time = time.time() - start_dec

            x_hat = F.pad(out_dec["x_hat"], unpad)

            # 检查解码结果是否有 NaN，跳过这些样本
            if not torch.isfinite(x_hat).all():
                nan_count += 1
                continue

            # 计算真实 bpfp
            num_points = x.size(0) * x.size(1) * x.size(2) * x.size(3)
            bpfp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_points if num_points > 0 else 0.0

            # 反变换到原始特征空间
            x_hat_np = x_hat.squeeze(0).squeeze(0).cpu().numpy()

            orig_shape_i = orig_shape[i].cpu().tolist() if orig_shape is not None else None

            if orig_shape_i is not None:
                shape = tuple(int(v) for v in orig_shape_i)
                x_hat_np = FeatureFolder.unpacking(x_hat_np, shape, model_type)
            norm_params_i = norm_params[i].cpu().numpy() if norm_params is not None else None
            trun_low = float(norm_params_i[0])
            trun_high = float(norm_params_i[1])
            x_hat_np = FeatureFolder.uniform_dequantization(x_hat_np, trun_low, trun_high, bit_depth)

            # MSE against original pre-truncation feature
            org_feat_i = org_feat[i].cpu().numpy() if torch.is_tensor(org_feat) else org_feat[i].numpy()
            mse = np.mean((x_hat_np - org_feat_i) ** 2)
            if not np.isfinite(mse):
                nan_count += 1
                continue

            bpfp_meter.update(bpfp)
            mse_meter.update(mse)
            enc_time_meter.update(enc_time)
            dec_time_meter.update(dec_time)

            # 分类
            x_hat_rec = torch.from_numpy(x_hat_np).to(device)
            label_i = label[i : i + 1].to(device)
            if x_hat_rec.dim() == 4:
                feat_hat = x_hat_rec.squeeze(1)
            elif x_hat_rec.dim() == 3:
                feat_hat = x_hat_rec
            else:
                feat_hat = x_hat_rec.unsqueeze(0)
            if feat_hat.dim() == 3 and feat_hat.shape[1] != 257 and feat_hat.shape[2] == 257:
                feat_hat = feat_hat.permute(0, 2, 1).contiguous()
            logits = dinowrapper.forward_from_tokens(feat_hat, layer_idx)
            pred = torch.max(logits, 1)[1]
            total_correct += (pred == label_i).sum().item()
            total_samples += 1

    acc = total_correct / total_samples if total_samples > 0 else 0.0

    results = {
        "bpfp": bpfp_meter.avg if bpfp_meter.count > 0 else 0.0,
        "mse": mse_meter.avg if mse_meter.count > 0 else 0.0,
        "acc": acc,
        "encoding_time_avg": enc_time_meter.avg if enc_time_meter.count > 0 else 0.0,
        "decoding_time_avg": dec_time_meter.avg if dec_time_meter.count > 0 else 0.0,
        "total_samples": total_samples,
        "nan_skipped": nan_count,
    }

    print(f"\nFinal Results:")
    print(f"  BPFP: {results['bpfp']:.4f}")
    print(f"  MSE: {results['mse']:.6f}")
    print(f"  Acc: {results['acc']:.4f}")
    print(f"  Avg Encoding Time: {results['encoding_time_avg']*1000:.2f} ms")
    print(f"  Avg Decoding Time: {results['decoding_time_avg']*1000:.2f} ms")
    if nan_count > 0:
        print(f"  WARNING: {nan_count} samples skipped due to NaN in decompress output")

    # 保存结果到 JSON
    if savepath:
        save_dir = Path(savepath).parent
        json_path = save_dir / "final_eval_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {json_path}")

    return results


@torch.no_grad()
def final_eval_seg(model, test_dataloader, dinowrapper, seg_head, layer_idx, model_type, bit_depth, savepath):
    """
    分割任务的最终评估（使用真实编解码 + slide inference 融合）
    """
    model.eval()
    device = next(model.parameters()).device

    if hasattr(model, "update"):
        model.update(force=True)

    bpfp_meter = AverageMeter()
    mse_meter = AverageMeter()
    enc_time_meter = AverageMeter()
    dec_time_meter = AverageMeter()
    nan_count = 0
    
    metric = IOUMetric(NUM_SEG_CLASSES, IGNORE_INDEX)

    print("\n" + "="*60)
    print("Final Evaluation with Real Entropy Coding (Segmentation)")
    print("="*60)

    for batch_idx, (d, gt, norm_params, orig_shape, crops, org_feat) in enumerate(test_dataloader):
        # batch_size=1: d [1,1,1+N,num_slides*D], gt [1,H,W], crops [1,num_slides,4]
        x = d.to(device)
        h, w = x.size(2), x.size(3)
        pad, unpad = compute_padding(h, w, min_div=2**6)
        x_padded = F.pad(x, pad, mode="constant", value=0)

        # 真实编码
        start_enc = time.time()
        out_enc = model.compress(x_padded)
        enc_time = time.time() - start_enc

        # 真实解码
        start_dec = time.time()
        out_dec = model.decompress(out_enc["strings"], out_enc["shape"])
        dec_time = time.time() - start_dec

        x_hat = F.pad(out_dec["x_hat"], unpad)

        if not torch.isfinite(x_hat).all():
            nan_count += 1
            continue

        # 计算真实 bpfp
        num_points = x.size(2) * x.size(3)
        bpfp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_points if num_points > 0 else 0.0

        # 反变换（squeeze 到单样本）
        x_hat_np = x_hat[0, 0].cpu().numpy()

        trun_low = float(norm_params[0, 0])
        trun_high = float(norm_params[0, 1])
        x_hat_np = FeatureFolder.uniform_dequantization(x_hat_np, trun_low, trun_high, bit_depth)

        # MSE against original pre-truncation feature
        org_feat_np = org_feat[0].cpu().numpy() if torch.is_tensor(org_feat) else org_feat[0].numpy()
        mse = np.mean((x_hat_np - org_feat_np) ** 2)
        if not np.isfinite(mse):
            nan_count += 1
            continue

        bpfp_meter.update(bpfp)
        mse_meter.update(mse)
        enc_time_meter.update(enc_time)
        dec_time_meter.update(dec_time)

        # unpacking + slide inference 分割评估
        gt_np = gt[0].cpu().numpy()
        orig_shape_i = tuple(int(v) for v in orig_shape[0].cpu().numpy())
        crops_np = crops[0].cpu().numpy()  # [num_slides, 4]
        
        feat_hat_3d = SegFeatureFolder.seg_unpacking(x_hat_np, orig_shape_i)
        pred = _slide_inference_seg(
            feat_hat_3d, gt_np, crops_np, dinowrapper, layer_idx, seg_head, device
        )
        metric.update(pred, gt_np)

    miou, acc, class_iou = metric.compute()

    results = {
        "bpfp": bpfp_meter.avg if bpfp_meter.count > 0 else 0.0,
        "mse": mse_meter.avg if mse_meter.count > 0 else 0.0,
        "miou": float(miou),
        "acc": float(acc),
        "encoding_time_avg": enc_time_meter.avg if enc_time_meter.count > 0 else 0.0,
        "decoding_time_avg": dec_time_meter.avg if dec_time_meter.count > 0 else 0.0,
        "total_samples": int(metric.confusion_matrix.sum()),
        "nan_skipped": nan_count,
    }

    print(f"\nFinal Results (Segmentation):")
    print(f"  BPFP: {results['bpfp']:.4f}")
    print(f"  MSE: {results['mse']:.6f}")
    print(f"  mIoU: {results['miou']:.4f}")
    print(f"  Acc: {results['acc']:.4f}")
    print(f"  Avg Encoding Time: {results['encoding_time_avg']*1000:.2f} ms")
    print(f"  Avg Decoding Time: {results['decoding_time_avg']*1000:.2f} ms")
    if nan_count > 0:
        print(f"  WARNING: {nan_count} samples skipped due to NaN in decompress output")

    if savepath:
        save_dir = Path(savepath).parent
        json_path = save_dir / "final_eval_seg_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {json_path}")

    return results


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument("-m", "--model", default="bmshj2018-factorized", choices=image_models.keys(), help="Model architecture (default: %(default)s)",
    )
    #gcs, model_type="sd3", task="tti", trun_flag=False, trun_low=-20, trun_high=20, quant_type="uniform", qsamples=0, bit_depth=1
    parser.add_argument("-model_type", 
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
        "--trun_flag",
        type=str,
        default="True",
        help="Truncation flag (True/False)",
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
        "-lr",
        "--learning-rate",
        default=1e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=32,
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
        "--aux-learning-rate",
        type=float,
        default=1e-3,
        help="Auxiliary loss learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(512, 512), #(hgt, wdt)
        help="Size of the patches to be cropped (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument(
        "--save", action="store_true", default=True, help="Save model to disk"
    )
    parser.add_argument("--seed", type=int, default=42, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    parser.add_argument("--train_split", type=str, default="train", help="Train split, e.g. train, train20k")
    parser.add_argument("--checkpoint", type=str, help="Path to a checkpoint")
    parser.add_argument("--gt_path", type=str, default=None, help="Path to test labels (base_name label) for classification")
    parser.add_argument("--layer", type=str, default="blk05", help="DINOv2 layer name, e.g. blk05")
    parser.add_argument("--log_dir", type=str, default="./runs/feature", help="TensorBoard log dir")
    # 分割任务相关参数
    parser.add_argument("--seg_test_root", type=str, default=None, help="Segmentation test feature root (e.g., features/voc2012_100)")
    parser.add_argument("--gt_root", type=str, default=None, help="VOC2012 SegmentationClass dir for segmentation GT")
    parser.add_argument("--seg_head_path", type=str, default=None, help="Path to segmentation head weights")
    parser.add_argument("--seg_precrop_root", type=str, default=None, help="Pre-cropped seg patches root (uses FeatureFolder for fast training I/O)")
    parser.add_argument("--classnames", type=str, default=os.path.join(_PROJECT_ROOT, "data", "imagenet", "classnames.txt"), help="Classnames file for CLIP zero-shot classification")
    args = parser.parse_args(argv)
    return args


def main(argv):
    args = parse_args(argv)
    args.trun_flag = args.trun_flag.lower() in ('true', '1', 'yes')

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    #gcs
    train_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [transforms.ToTensor()]
    )

    #gcs 
    if args.task == "seg":
        seg_test_root = args.seg_test_root if args.seg_test_root else args.dataset
        if args.seg_precrop_root:
            print(f"  使用预裁剪数据训练: {args.seg_precrop_root}")
            train_dataset = FeatureFolder(
                args.seg_precrop_root, split="train",
                transform=train_transforms, model_type=args.model_type,
                layer=args.layer, task=args.task,
                trun_flag=args.trun_flag, trun_low=args.trun_low, trun_high=args.trun_high,
                quant_type=args.quant_type, qsamples=args.qsamples,
                bit_depth=args.bit_depth, patch_size=None,
            )
        else:
            train_dataset = SegFeatureFolder(
                args.dataset, 
                split="train",
                transform=train_transforms, 
                model_type=args.model_type, 
                layer=args.layer, 
                task=args.task,
                trun_flag=args.trun_flag, 
                trun_low=args.trun_low, 
                trun_high=args.trun_high, 
                quant_type=args.quant_type,
                qsamples=args.qsamples,
                bit_depth=args.bit_depth, 
                patch_size=args.patch_size,
                gt_root=None,
            )
        test_dataset = SegFeatureFolder(
            seg_test_root, 
            split="test",
            transform=test_transforms, 
            model_type=args.model_type, 
            layer=args.layer,
            task=args.task, 
            trun_flag=args.trun_flag, 
            trun_low=args.trun_low, 
            trun_high=args.trun_high, 
            quant_type=args.quant_type,
            qsamples=args.qsamples,
            bit_depth=args.bit_depth, 
            patch_size=None,  # 测试时不裁剪
            gt_root=args.gt_root  # 测试时需要 GT
        )
    else:
        # 分类任务：使用原有的 FeatureFolder
        train_dataset = FeatureFolder(args.dataset, split=args.train_split, transform=train_transforms, model_type=args.model_type, layer=args.layer, task=args.task, trun_flag=args.trun_flag, trun_low=args.trun_low, trun_high=args.trun_high, quant_type=args.quant_type, qsamples=args.qsamples, bit_depth=args.bit_depth, patch_size=args.patch_size)
        test_dataset = FeatureFolder(args.dataset, split="test", transform=test_transforms, model_type=args.model_type, layer=args.layer, task=args.task, trun_flag=args.trun_flag, trun_low=args.trun_low, trun_high=args.trun_high, quant_type=args.quant_type, qsamples=args.qsamples, bit_depth=args.bit_depth, patch_size=args.patch_size, gt_path=args.gt_path)

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        persistent_workers=True if args.task == "seg" else False,
        prefetch_factor=8 if args.task == "seg" else 2,
        pin_memory=(device == "cuda"),
    )

    test_batch_size = 1 if args.task == "seg" else min(8, args.batch_size)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        num_workers=min(args.num_workers, test_batch_size),
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    net = image_models[args.model](quality=1)   # set default quality level to 1
    # net = image_models[args.model](quality=3)
    net = net.to(device)
    
    is_clip = args.model_type.startswith("clip")
    is_dinov2 = args.model_type.startswith("dinov2")
    if args.task == "seg":
        dinowrapper = Dinov2Wrapper(head_layers=1, model_name=args.model_type, device=device)
        seg_head_path = args.seg_head_path
        dinowrapper.load_segmentation_head(seg_head_path)
        seg_head = dinowrapper.seg_head
    elif is_clip:
        dinowrapper = ClipWrapper(args.classnames, device=device)
        seg_head = None
    elif is_dinov2:
        dinowrapper = Dinov2Wrapper(head_layers=1, model_name=args.model_type, device=device)
        seg_head = None
    else:
        dinowrapper = Dinov2Wrapper(head_layers=1, device=device)
        seg_head = None

    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint and os.path.exists(args.checkpoint):  # load from previous checkpoint
        print("Loading", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint["epoch"] + 1
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

    best_loss = float("inf")
    layer_digits = "".join([c for c in args.layer if c.isdigit()])
    layer_idx = int(layer_digits) if layer_digits else 0

    writer = SummaryWriter(args.log_dir)

    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
            writer,
        )
        run_downstream_eval = (epoch % 50 == 0) or (epoch == args.epochs - 1)
        if args.task == "seg":
            loss = test_epoch_seg(epoch, test_dataloader, net, dinowrapper, seg_head, layer_idx, criterion, args.model_type, args.bit_depth, writer, run_seg_eval=run_downstream_eval)
        else:
            loss = test_epoch(epoch, test_dataloader, net, dinowrapper, criterion, layer_idx, args.model_type, args.bit_depth, writer, run_cls_eval=run_downstream_eval)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if epoch % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()
        
        if args.save:
            # 每个 epoch 检查是否是 best，如果是就保存 best checkpoint
            if is_best:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best=True,
                    filename=args.savepath,
                )
            # 每 50 个 epoch 保存常规 checkpoint
            if (epoch + 1) % 50 == 0:
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": net.state_dict(),
                        "loss": loss,
                        "optimizer": optimizer.state_dict(),
                        "aux_optimizer": aux_optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                    },
                    is_best=False,  # 常规保存不覆盖 best
                    filename=args.savepath,
                )

    if writer is not None:
        writer.close()

    # ========== 训练完成后，加载 best checkpoint 进行最终真实编码评估 ==========
    if args.savepath:
        best_ckpt_path = get_best_checkpoint_path(args.savepath)
        if os.path.exists(best_ckpt_path):
            print(f"\n{'='*60}")
            print(f"Loading best checkpoint: {best_ckpt_path}")
            print(f"{'='*60}")
            
            # 使用 from_state_dict 加载模型，确保架构参数匹配（对齐 eval_model）
            checkpoint = torch.load(best_ckpt_path, map_location=device)
            state_dict = checkpoint["state_dict"]
            
            # 获取模型类并从 state_dict 推断架构（使用 model_architectures 而非 image_models）
            model_cls = model_architectures[args.model]
            net_eval = model_cls.from_state_dict(state_dict)
            net_eval = net_eval.to(device)
            net_eval.eval()
            
            # 进行最终评估（final_eval 内部会调用 model.update）
            if args.task == "seg":
                final_eval_seg(
                    net_eval, 
                    test_dataloader, 
                    dinowrapper,
                    seg_head, 
                    layer_idx,
                    args.model_type, 
                    args.bit_depth, 
                    args.savepath
                )
            else:
                final_eval(
                    net_eval, 
                    test_dataloader, 
                    dinowrapper, 
                    layer_idx, 
                    args.model_type, 
                    args.bit_depth, 
                    args.savepath
                )
        else:
            print(f"\nWarning: Best checkpoint not found at {best_ckpt_path}")
            print("Skipping final evaluation.")


if __name__ == "__main__":
    main(sys.argv[1:])
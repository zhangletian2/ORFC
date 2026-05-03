"""dtufc reproduction: kmeans-preprocessed hyperprior training with ORFC evaluation.

Train on offline kmeans-quantized crops [256,256].
test_epoch: dtufc-style R-D loss on [256,256] test crops (quantized domain).
cls_eval_epoch: full [257,1024] → kmeans dequant → DINOv2 classification (every N epochs).
"""

import argparse
import gc
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from PIL import Image

from compressai.losses import RateDistortionLoss
from compressai.ops import compute_padding
from compressai.optimizers import net_aux_optimizer
from compressai.zoo import image_models
from compressai.zoo.image import model_architectures

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
sys.path.append(os.path.join(_PROJECT_ROOT, "coding", "orfc"))
from backbone.wrapper import Dinov2Wrapper

import warnings
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

# ---------------------------------------------------------------------------
# Segmentation constants
# ---------------------------------------------------------------------------

NUM_SEG_CLASSES = 21
IGNORE_INDEX = 255
SEG_CROP_SIZE = (512, 512)
SEG_STRIDE = (341, 341)
SEG_PATCH_SIZE = 14
SEG_IMG_SCALE = (2048, 512)


class IOUMetric:
    """Accumulator for mIoU computation."""

    def __init__(self, num_classes, ignore_index=255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, pred, target):
        mask = target != self.ignore_index
        pred = pred[mask].astype(np.int64)
        target = target[mask].astype(np.int64)
        indices = self.num_classes * target + pred
        cm = np.bincount(indices, minlength=self.num_classes ** 2)
        self.confusion_matrix += cm.reshape(self.num_classes, self.num_classes)

    def compute(self):
        intersection = np.diag(self.confusion_matrix)
        union = (self.confusion_matrix.sum(axis=1) +
                 self.confusion_matrix.sum(axis=0) - intersection)
        iou = intersection / (union + 1e-10)
        valid = union > 0
        miou = iou[valid].mean()
        acc = intersection.sum() / (self.confusion_matrix.sum() + 1e-10)
        return miou, acc, iou


def compute_resized_shape(orig_h, orig_w, img_scale=SEG_IMG_SCALE):
    """Reproduce mmseg Resize(keep_ratio=True) scaling logic."""
    max_long = max(img_scale)
    max_short = min(img_scale)
    scale_factor = min(max_long / max(orig_h, orig_w),
                       max_short / min(orig_h, orig_w))
    return int(round(orig_h * scale_factor)), int(round(orig_w * scale_factor))


def get_slide_crops(h_img, w_img, crop_size=SEG_CROP_SIZE, stride=SEG_STRIDE):
    """Compute slide window crop regions, returns list of (y1, x1, y2, x2)."""
    h_crop, w_crop = crop_size
    h_stride, w_stride = stride
    crops = []
    for h_idx in range(0, max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1):
        for w_idx in range(0, max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)
            crops.append((y1, x1, y2, x2))
    return crops


def seg_unpacking(feat_2d, orig_shape):
    """Inverse packing: [1+N, num_slides*D] -> [num_slides, 1+N, D]."""
    if len(orig_shape) != 3:
        return feat_2d
    num_slides, seq_len, dim = orig_shape
    slides = np.split(feat_2d, num_slides, axis=-1)
    return np.stack(slides, axis=0)

# ---------------------------------------------------------------------------
# Dataset: load pre-processed kmeans-quantized features
# ---------------------------------------------------------------------------

class PreprocessedTrainFolder(Dataset):
    """Load offline kmeans-quantized crops for training.

    Supports two modes:
      - Packed: {root}/train_all.npy exists -> mmap load [N, 256, 256]
      - Loose:  {root}/train/*.npy -> load individual files
    """

    def __init__(self, root):
        packed = Path(root) / "train_all.npy"
        if packed.is_file():
            self.data = np.load(str(packed), mmap_mode='r')
            self.samples = None
            print(f"  Train: mmap packed {packed} -> {self.data.shape}")
        else:
            train_dir = Path(root) / "train"
            if not train_dir.is_dir():
                raise RuntimeError(f"Train dir not found: {train_dir}")
            self.samples = sorted(f for f in train_dir.iterdir() if f.suffix == ".npy")
            self.data = None

    def __getitem__(self, index):
        if self.data is not None:
            feat = self.data[index].astype(np.float32)
        else:
            feat = np.load(self.samples[index]).astype(np.float32)
        return np.expand_dims(feat, axis=0)

    def __len__(self):
        return len(self.data) if self.data is not None else len(self.samples)


class PreprocessedTestCropFolder(Dataset):
    """Load offline kmeans-quantized TEST crops (same format as train).

    Supports two modes:
      - Packed: {root}/test_crop_all.npy exists -> mmap load
      - Loose:  {root}/test_crop/*.npy -> load individual files
    """

    def __init__(self, root):
        packed = Path(root) / "test_crop_all.npy"
        if packed.is_file():
            self.data = np.load(str(packed), mmap_mode='r')
            self.samples = None
            print(f"  Test crop: mmap packed {packed} -> {self.data.shape}")
        else:
            test_dir = Path(root) / "test_crop"
            if not test_dir.is_dir():
                raise RuntimeError(f"Test crop dir not found: {test_dir}")
            self.samples = sorted(f for f in test_dir.iterdir() if f.suffix == ".npy")
            self.data = None

    def __getitem__(self, index):
        if self.data is not None:
            feat = self.data[index].astype(np.float32)
        else:
            feat = np.load(self.samples[index]).astype(np.float32)
        return np.expand_dims(feat, axis=0)

    def __len__(self):
        return len(self.data) if self.data is not None else len(self.samples)


class PreprocessedTestFolder(Dataset):
    """Load kmeans-quantized FULL test features + original features + labels.

    Quantized: {quant_root}/test/*.npy  → [257, 1024] float32 in [0,1]
    Original:  {orig_root}/test/{model}/{layer}/*.npy → [257, 1024] float32
    Labels:    gt_path text file (stem → class_idx)
    Used for classification evaluation every N epochs.
    """

    def __init__(self, quant_root, orig_root, model_type, layer, gt_path,
                 trun_low, trun_high):
        quant_dir = Path(quant_root) / "test"
        orig_dir = Path(orig_root) / "test" / model_type / layer

        if not quant_dir.is_dir():
            raise RuntimeError(f"Quantized test dir not found: {quant_dir}")
        if not orig_dir.is_dir():
            raise RuntimeError(f"Original test dir not found: {orig_dir}")

        self.quant_samples = sorted(f for f in quant_dir.iterdir() if f.suffix == ".npy")
        self._orig_dir = orig_dir
        self.trun_low = trun_low
        self.trun_high = trun_high

        self.gt = {}
        if gt_path:
            with open(gt_path) as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    base, idx = ln.split()
                    self.gt[base] = int(idx)

    def __getitem__(self, index):
        qpath = self.quant_samples[index]
        stem = qpath.stem

        quant_feat = np.load(qpath).astype(np.float32)
        quant_feat = np.expand_dims(quant_feat, axis=0)  # [1, 257, 1024]

        orig_path = self._orig_dir / f"{stem}.npy"
        org_feat = np.load(orig_path).astype(np.float32)  # [257, 1024]

        label = self.gt.get(stem, -1)
        orig_shape = np.array(org_feat.shape, dtype=np.int64)
        norm_params = np.array([self.trun_low, self.trun_high], dtype=np.float32)

        return quant_feat, label, norm_params, orig_shape, org_feat

    def __len__(self):
        return len(self.quant_samples)


class PreprocessedSegTestFolder(Dataset):
    """Load kmeans-quantized packed seg features + original features + GT masks.

    Quantized: {quant_root}/test/*.npy  -> [1+N, num_slides*D] float32 in [0,1]
    Original:  {orig_root}/{model}/{layer}/*.npy -> [num_slides, 1+N, D] float32
    GT:        {gt_root}/*.png -> [H, W] uint8
    Used for segmentation mIoU evaluation every N epochs.
    """

    def __init__(self, quant_root, orig_root, model_type, layer, gt_root,
                 trun_low, trun_high):
        quant_dir = Path(quant_root) / "test"
        orig_dir = Path(orig_root) / model_type / layer

        if not quant_dir.is_dir():
            raise RuntimeError(f"Quantized seg test dir not found: {quant_dir}")
        if not orig_dir.is_dir():
            raise RuntimeError(f"Original seg test dir not found: {orig_dir}")

        self.quant_samples = sorted(f for f in quant_dir.iterdir() if f.suffix == ".npy")
        self._orig_dir = orig_dir
        self._gt_root = Path(gt_root)
        self.trun_low = trun_low
        self.trun_high = trun_high

    def __getitem__(self, index):
        qpath = self.quant_samples[index]
        stem = qpath.stem

        quant_feat = np.load(qpath).astype(np.float32)
        quant_feat = np.expand_dims(quant_feat, axis=0)  # [1, 1+N, num_slides*D]

        orig_path = self._orig_dir / f"{stem}.npy"
        org_feat_3d = np.load(orig_path).astype(np.float32)  # [num_slides, 1+N, D]
        orig_shape = np.array(org_feat_3d.shape, dtype=np.int64)

        num_slides = org_feat_3d.shape[0]
        org_feat_packed = np.concatenate(
            [org_feat_3d[s] for s in range(num_slides)], axis=-1
        )  # [1+N, num_slides*D]

        gt_path = self._gt_root / f"{stem}.png"
        gt = np.array(Image.open(gt_path))  # [H, W]

        orig_h, orig_w = gt.shape[:2]
        h_img, w_img = compute_resized_shape(orig_h, orig_w)
        crops = np.array(get_slide_crops(h_img, w_img), dtype=np.int64)

        norm_params = np.array([self.trun_low, self.trun_high], dtype=np.float32)

        return quant_feat, gt, norm_params, orig_shape, crops, org_feat_packed

    def __len__(self):
        return len(self.quant_samples)


# ---------------------------------------------------------------------------
# kmeans dequantization
# ---------------------------------------------------------------------------

def load_kmeans_centers(mapping_path):
    """Load centers from mapping JSON (legacy list or v2 dict)."""
    with open(mapping_path) as f:
        payload = json.load(f)
    if isinstance(payload, list):
        centers = np.array(payload, dtype=np.float64)
    elif isinstance(payload, dict):
        centers = np.array(payload.get("centers", payload.get("quantization_points", [])),
                           dtype=np.float64)
    else:
        raise ValueError(f"Unsupported mapping format in {mapping_path}")
    return np.sort(centers)


def kmeans_dequantize_np(x_hat_np, centers, bit_depth):
    """Dequantize model output using kmeans centers lookup.

    Matches dtufc original: idx = round(x * 2^bit_depth), clipped to [0, 2^bd-1].
    x_hat_np: float array (any shape), values approximately in [0, 1)
    Returns: dequantized float32 array (same shape)
    """
    max_idx = 2 ** bit_depth - 1
    idx = np.clip(np.round(x_hat_np * (2 ** bit_depth)), 0, max_idx).astype(np.int32)
    idx = np.clip(idx, 0, len(centers) - 1)
    return centers[idx].astype(np.float32)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

class AverageMeter:
    def __init__(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(model, criterion, dataloader, optimizer, aux_optimizer,
                    epoch, clip_max_norm, writer):
    model.train()
    device = next(model.parameters()).device
    total_loss = total_bpp = total_mse = total_aux = 0.0
    n = 0
    for d in dataloader:
        d = d.to(device)
        optimizer.zero_grad()
        aux_optimizer.zero_grad()
        out = model(d)
        out_c = criterion(out, d)
        out_c["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()
        aux = model.aux_loss()
        aux.backward()
        aux_optimizer.step()
        total_loss += out_c["loss"].detach()
        total_bpp += out_c["bpp_loss"].detach()
        total_mse += out_c["mse_loss"].detach()
        total_aux += aux.detach()
        n += 1

    avg = lambda t: (t / n).item()
    print(f"Train epoch {epoch}: Loss: {avg(total_loss):.3f} | "
          f"MSE: {avg(total_mse):.4f} | Bpp: {avg(total_bpp):.4f} | Aux: {avg(total_aux):.2f}")
    if writer:
        writer.add_scalar("train/loss", avg(total_loss), epoch)
        writer.add_scalar("train/mse_loss", avg(total_mse), epoch)
        writer.add_scalar("train/bpp_loss", avg(total_bpp), epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)


def test_epoch(epoch, test_dataloader, model, criterion):
    """dtufc-style test: R-D loss on [256,256] test crops (quantized domain).

    Identical to the original dtufc test_epoch — no dequantization, no cls eval.
    Used for best checkpoint selection and LR scheduling.
    """
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    aux_loss = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:
            d = d.to(device)
            out_net = model(d)
            out_criterion = criterion(out_net, d)

            aux_loss.update(model.aux_loss())
            bpp_loss.update(out_criterion["bpp_loss"])
            loss.update(out_criterion["loss"])
            mse_loss.update(out_criterion["mse_loss"])

    print(
        f"Test epoch {epoch}: Average losses:"
        f"\tLoss: {loss.avg:.7f} |"
        f"\tMSE loss: {mse_loss.avg:.7f} |"
        f"\tBpp loss: {bpp_loss.avg:.7f} |"
        f"\tAux loss: {aux_loss.avg:.4f}\n"
    )
    if isinstance(loss.avg, torch.Tensor):
        loss_val = loss.avg.item()
    else:
        loss_val = float(loss.avg)
    return loss_val


def cls_eval_epoch(epoch, dataloader, model, dinowrapper, criterion, layer_idx,
                   centers, bit_depth, writer):
    """Full [257,1024] evaluation: kmeans dequant → MSE vs original → DINOv2 cls.

    Run periodically (e.g., every 50 epochs) to monitor downstream task accuracy.
    """
    model.eval()
    device = next(model.parameters()).device
    loss_m = AverageMeter()
    bpp_m = AverageMeter()
    mse_m = AverageMeter()
    total_correct = total_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            d, label, norm_params, orig_shape, org_feat = batch
            d = d.to(device)

            h, w = d.size(2), d.size(3)
            pad, unpad = compute_padding(h, w, min_div=2 ** 6)
            out = model(F.pad(d, pad, mode="constant", value=0))
            x_hat = F.pad(out["x_hat"], unpad)

            num_pts = d.size(0) * d.size(1) * d.size(2) * d.size(3)
            bpp = sum(-lh.clamp(min=1e-9).log2().sum().item()
                      for lh in out["likelihoods"].values()) / max(num_pts, 1)

            x_hat_np = x_hat.squeeze(1).cpu().numpy()
            x_hat_np = kmeans_dequantize_np(x_hat_np, centers, bit_depth)

            org_np = org_feat.numpy() if not torch.is_tensor(org_feat) else org_feat.cpu().numpy()
            mse = np.mean((x_hat_np - org_np) ** 2)
            rd = criterion.lmbda * (255 ** 2) * mse + bpp

            loss_m.update(rd)
            bpp_m.update(bpp)
            mse_m.update(mse)

            label = label.to(device)
            feat_hat = torch.from_numpy(x_hat_np).to(device)
            if feat_hat.dim() == 3 and feat_hat.shape[1] != 257 and feat_hat.shape[2] == 257:
                feat_hat = feat_hat.permute(0, 2, 1).contiguous()
            logits = dinowrapper.forward_from_tokens(feat_hat, layer_idx)
            pred = logits.argmax(1)
            total_correct += (pred == label).sum().item()
            total_samples += label.size(0)

    acc = total_correct / total_samples if total_samples > 0 else 0.0
    print(f"CLS eval epoch {epoch}: Loss: {loss_m.avg:.3f} | MSE: {mse_m.avg:.4f} | "
          f"BPFP: {bpp_m.avg:.4f} | Acc: {acc:.4f}\n")
    if writer:
        writer.add_scalar("cls/loss", loss_m.avg, epoch)
        writer.add_scalar("cls/mse", mse_m.avg, epoch)
        writer.add_scalar("cls/bpfp", bpp_m.avg, epoch)
        writer.add_scalar("cls/acc", acc, epoch)
    return acc


def _slide_inference_seg(feat_hat_3d, gt_np, crops, dinowrapper, layer_idx, device):
    """Single-image slide inference segmentation evaluation.

    Args:
        feat_hat_3d: numpy [num_slides, 1+N, D] decoded original-space features
        gt_np: numpy [H_gt, W_gt] GT segmentation mask
        crops: numpy [num_slides, 4] pre-computed slide coords (y1, x1, y2, x2)
        dinowrapper: Dinov2Wrapper (must have seg_head loaded)
        layer_idx: starting block index
        device: computation device
    Returns:
        pred: numpy [H_gt, W_gt] predicted segmentation map
    """
    h_crop, w_crop = SEG_CROP_SIZE
    orig_h, orig_w = gt_np.shape[:2]
    h_img, w_img = compute_resized_shape(orig_h, orig_w)

    num_slides = feat_hat_3d.shape[0]
    assert num_slides == len(crops), (
        f"slides mismatch: feat={num_slides} vs crops={len(crops)}")

    preds = torch.zeros((1, NUM_SEG_CLASSES, h_img, w_img), device=device)
    count_mat = torch.zeros((1, 1, h_img, w_img), device=device)

    for s, (y1, x1, y2, x2) in enumerate(crops):
        tokens = torch.from_numpy(feat_hat_3d[s]).float().to(device).unsqueeze(0)

        actual_h = y2 - y1
        actual_w = x2 - x1
        padded_h = math.ceil(actual_h / SEG_PATCH_SIZE) * SEG_PATCH_SIZE
        padded_w = math.ceil(actual_w / SEG_PATCH_SIZE) * SEG_PATCH_SIZE
        feat_h = padded_h // SEG_PATCH_SIZE
        feat_w = padded_w // SEG_PATCH_SIZE

        logits = dinowrapper.forward_from_tokens_seg(tokens, layer_idx, feat_h, feat_w)
        logits_up = F.interpolate(logits, size=(h_crop, w_crop),
                                  mode='bilinear', align_corners=False)
        logits_crop = logits_up[:, :, :actual_h, :actual_w]

        preds[:, :, y1:y2, x1:x2] += logits_crop
        count_mat[:, :, y1:y2, x1:x2] += 1

    assert (count_mat == 0).sum() == 0, "count_mat has zeros"
    preds = preds / count_mat

    if (h_img, w_img) != (orig_h, orig_w):
        preds = F.interpolate(preds, size=(orig_h, orig_w),
                              mode='bilinear', align_corners=False)
    return preds.argmax(dim=1).squeeze(0).cpu().numpy()


def seg_eval_epoch(epoch, dataloader, model, dinowrapper, criterion, layer_idx,
                   centers, bit_depth, writer):
    """Full seg evaluation: kmeans dequant -> unpack -> slide inference -> mIoU.

    batch_size must be 1 (variable spatial sizes across samples).
    """
    model.eval()
    device = next(model.parameters()).device
    loss_m = AverageMeter()
    bpp_m = AverageMeter()
    mse_m = AverageMeter()
    metric = IOUMetric(NUM_SEG_CLASSES, IGNORE_INDEX)

    with torch.no_grad():
        for batch in dataloader:
            d, gt, norm_params, orig_shape, crops, org_feat = batch
            d = d.to(device)

            h, w = d.size(2), d.size(3)
            pad, unpad = compute_padding(h, w, min_div=2 ** 6)
            out = model(F.pad(d, pad, mode="constant", value=0))
            x_hat = F.pad(out["x_hat"], unpad)

            num_pts = d.size(2) * d.size(3)
            bpp = sum(-lh.clamp(min=1e-9).log2().sum().item()
                      for lh in out["likelihoods"].values()) / max(num_pts, 1)

            x_hat_np = x_hat[0, 0].cpu().numpy()
            x_hat_np = kmeans_dequantize_np(x_hat_np, centers, bit_depth)

            org_np = org_feat[0].cpu().numpy() if torch.is_tensor(org_feat) \
                else org_feat[0].numpy()
            mse = np.mean((x_hat_np - org_np) ** 2)
            rd = criterion.lmbda * (255 ** 2) * mse + bpp

            loss_m.update(rd)
            bpp_m.update(bpp)
            mse_m.update(mse)

            orig_shape_i = tuple(int(v) for v in orig_shape[0].cpu().numpy())
            feat_hat_3d = seg_unpacking(x_hat_np, orig_shape_i)

            gt_np = gt[0].cpu().numpy()
            crops_np = crops[0].cpu().numpy()
            pred = _slide_inference_seg(feat_hat_3d, gt_np, crops_np,
                                        dinowrapper, layer_idx, device)
            metric.update(pred, gt_np)

    miou, acc, class_iou = metric.compute()
    print(f"SEG eval epoch {epoch}: Loss: {loss_m.avg:.3f} | MSE: {mse_m.avg:.4f} | "
          f"BPFP: {bpp_m.avg:.4f} | mIoU: {miou:.4f}\n")
    if writer:
        writer.add_scalar("seg/loss", loss_m.avg, epoch)
        writer.add_scalar("seg/mse", mse_m.avg, epoch)
        writer.add_scalar("seg/bpfp", bpp_m.avg, epoch)
        writer.add_scalar("seg/miou", miou, epoch)
    return miou


@torch.no_grad()
def final_eval_seg(model, dataloader, dinowrapper, layer_idx, centers, bit_depth,
                   savepath):
    """Real entropy coding evaluation for segmentation task."""
    model.eval()
    device = next(model.parameters()).device
    if hasattr(model, "update"):
        model.update(force=True)

    bpfp_m = AverageMeter()
    mse_m = AverageMeter()
    enc_t = AverageMeter()
    dec_t = AverageMeter()
    metric = IOUMetric(NUM_SEG_CLASSES, IGNORE_INDEX)
    nan_count = 0

    print("\n" + "=" * 60)
    print("Final Evaluation with Real Entropy Coding (Segmentation)")
    print("=" * 60)

    for batch in dataloader:
        d, gt, norm_params, orig_shape, crops, org_feat = batch
        x = d.to(device)
        h, w = x.size(2), x.size(3)
        pad, unpad = compute_padding(h, w, min_div=2 ** 6)
        x_padded = F.pad(x, pad, mode="constant", value=0)

        t0 = time.time()
        enc = model.compress(x_padded)
        et = time.time() - t0

        t0 = time.time()
        dec = model.decompress(enc["strings"], enc["shape"])
        dt = time.time() - t0

        x_hat = F.pad(dec["x_hat"], unpad)
        if not torch.isfinite(x_hat).all():
            nan_count += 1
            continue

        num_pts = x.size(2) * x.size(3)
        bpfp = sum(len(s[0]) for s in enc["strings"]) * 8.0 / max(num_pts, 1)

        x_hat_np = x_hat[0, 0].cpu().numpy()
        x_hat_np = kmeans_dequantize_np(x_hat_np, centers, bit_depth)

        org_np = org_feat[0].cpu().numpy() if torch.is_tensor(org_feat) \
            else org_feat[0].numpy()
        mse = float(np.mean((x_hat_np - org_np) ** 2))
        if not np.isfinite(mse):
            nan_count += 1
            continue

        bpfp_m.update(bpfp)
        mse_m.update(mse)
        enc_t.update(et)
        dec_t.update(dt)

        orig_shape_i = tuple(int(v) for v in orig_shape[0].cpu().numpy())
        feat_hat_3d = seg_unpacking(x_hat_np, orig_shape_i)

        gt_np = gt[0].cpu().numpy()
        crops_np = crops[0].cpu().numpy()
        pred = _slide_inference_seg(feat_hat_3d, gt_np, crops_np,
                                    dinowrapper, layer_idx, device)
        metric.update(pred, gt_np)

    miou, acc, class_iou = metric.compute()
    results = {
        "bpfp": bpfp_m.avg if bpfp_m.count else 0,
        "mse": mse_m.avg if mse_m.count else 0,
        "miou": float(miou),
        "acc": float(acc),
        "enc_time_ms": (enc_t.avg * 1000) if enc_t.count else 0,
        "dec_time_ms": (dec_t.avg * 1000) if dec_t.count else 0,
        "total_samples": int(metric.confusion_matrix.sum()),
        "nan_skipped": nan_count,
    }
    print(f"\nFinal (Seg): BPFP={results['bpfp']:.4f}  MSE={results['mse']:.6f}  "
          f"mIoU={results['miou']:.4f}  Acc={results['acc']:.4f}  "
          f"EncT={results['enc_time_ms']:.1f}ms  DecT={results['dec_time_ms']:.1f}ms")
    if nan_count:
        print(f"  WARNING: {nan_count} samples skipped (NaN)")

    if savepath:
        json_path = Path(savepath).parent / "final_eval_seg_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {json_path}")
    return results


@torch.no_grad()
def final_eval(model, dataloader, dinowrapper, layer_idx, centers, bit_depth, savepath):
    """Real entropy coding evaluation."""
    model.eval()
    device = next(model.parameters()).device
    if hasattr(model, "update"):
        model.update(force=True)

    bpfp_m = AverageMeter()
    mse_m = AverageMeter()
    enc_t = AverageMeter()
    dec_t = AverageMeter()
    total_correct = total_samples = nan_count = 0

    print("\n" + "=" * 60)
    print("Final Evaluation with Real Entropy Coding")
    print("=" * 60)

    for batch in dataloader:
        d, label, norm_params, orig_shape, org_feat = batch
        d = d.to(device)
        bs = d.size(0)

        for i in range(bs):
            x = d[i:i + 1]
            h, w = x.size(2), x.size(3)
            pad, unpad = compute_padding(h, w, min_div=2 ** 6)
            x_padded = F.pad(x, pad, mode="constant", value=0)

            t0 = time.time()
            enc = model.compress(x_padded)
            et = time.time() - t0

            t0 = time.time()
            dec = model.decompress(enc["strings"], enc["shape"])
            dt = time.time() - t0

            x_hat = F.pad(dec["x_hat"], unpad)
            if not torch.isfinite(x_hat).all():
                nan_count += 1
                continue

            num_pts = x.numel()
            bpfp = sum(len(s[0]) for s in enc["strings"]) * 8.0 / max(num_pts, 1)

            x_hat_np = x_hat.squeeze(0).squeeze(0).cpu().numpy()
            x_hat_np = kmeans_dequantize_np(x_hat_np, centers, bit_depth)

            org_np = org_feat[i].numpy() if not torch.is_tensor(org_feat) else org_feat[i].cpu().numpy()
            mse = float(np.mean((x_hat_np - org_np) ** 2))
            if not np.isfinite(mse):
                nan_count += 1
                continue

            bpfp_m.update(bpfp)
            mse_m.update(mse)
            enc_t.update(et)
            dec_t.update(dt)

            feat_hat = torch.from_numpy(x_hat_np).to(device).unsqueeze(0)
            label_i = label[i:i + 1].to(device)
            logits = dinowrapper.forward_from_tokens(feat_hat, layer_idx)
            total_correct += (logits.argmax(1) == label_i).sum().item()
            total_samples += 1

    acc = total_correct / max(total_samples, 1)
    results = {
        "bpfp": bpfp_m.avg if bpfp_m.count else 0,
        "mse": mse_m.avg if mse_m.count else 0,
        "acc": acc,
        "enc_time_ms": (enc_t.avg * 1000) if enc_t.count else 0,
        "dec_time_ms": (dec_t.avg * 1000) if dec_t.count else 0,
        "total_samples": total_samples,
        "nan_skipped": nan_count,
    }
    print(f"\nFinal: BPFP={results['bpfp']:.4f}  MSE={results['mse']:.6f}  "
          f"Acc={results['acc']:.4f}  EncT={results['enc_time_ms']:.1f}ms  "
          f"DecT={results['dec_time_ms']:.1f}ms")
    if nan_count:
        print(f"  WARNING: {nan_count} samples skipped (NaN)")

    if savepath:
        json_path = Path(savepath).parent / "final_eval_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  Saved: {json_path}")
    return results


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def get_best_path(filename):
    p = Path(filename)
    stem = p.stem
    if stem.endswith(".pth"):
        stem = stem[:-4]
    return str(p.parent / f"{stem}_best.pth.tar")


def save_checkpoint(state, is_best, filename):
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, get_best_path(filename))


def save_cls_checkpoint(state, epoch, is_best_cls, filename):
    """Save checkpoint at CLS eval point + best-CLS-accuracy checkpoint."""
    parent = Path(filename).parent
    parent.mkdir(parents=True, exist_ok=True)
    ep_path = parent / f"checkpoint_cls_ep{epoch}.pth.tar"
    torch.save(state, ep_path)
    if is_best_cls:
        best_cls_path = parent / "checkpoint_cls_best.pth.tar"
        shutil.copyfile(str(ep_path), str(best_cls_path))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv):
    p = argparse.ArgumentParser(description="dtufc reproduction training.")
    p.add_argument("-m", "--model", default="bmshj2018-hyperprior",
                   choices=image_models.keys())
    p.add_argument("--model_type", type=str, default="dinov2_vitl14")
    p.add_argument("--layer", type=str, default="blk05")
    p.add_argument("--task", type=str, default="cls", choices=["cls", "seg"],
                   help="Downstream task: cls (classification) or seg (segmentation)")
    p.add_argument("--trun_low", type=float, default=-1)
    p.add_argument("--trun_high", type=float, default=1)
    p.add_argument("--bit_depth", type=int, default=8)
    p.add_argument("--mapping", type=str, required=True,
                   help="Path to kmeans mapping JSON")
    p.add_argument("--train_data", type=str, required=True,
                   help="Pre-processed train data root (contains train/ subdir)")
    p.add_argument("--test_data", type=str, default=None,
                   help="Original features root for cls (contains test/{model}/{layer}/)")
    p.add_argument("--gt_path", type=str, default=None,
                   help="Classification labels file (stem label)")
    # segmentation-specific args
    p.add_argument("--seg_test_root", type=str, default=None,
                   help="Seg test feature root (contains {model}/{layer}/*.npy)")
    p.add_argument("--gt_root", type=str, default=None,
                   help="VOC2012 SegmentationClass dir for seg GT masks")
    p.add_argument("--seg_head_path", type=str, default=None,
                   help="Path to pre-trained segmentation head weights")
    p.add_argument("-mp", "--savepath", type=str)
    p.add_argument("--log_dir", type=str, default="./runs/dtufc")
    p.add_argument("-e", "--epochs", type=int, default=500)
    p.add_argument("-lr", "--learning-rate", type=float, default=1e-4)
    p.add_argument("--aux-learning-rate", type=float, default=1e-3)
    p.add_argument("-n", "--num-workers", type=int, default=8)
    p.add_argument("--lambda", dest="lmbda", type=float, default=1e-2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--test-batch-size", dest="test_batch_size", type=int, default=64)
    p.add_argument("--patch-size", type=int, nargs=2, default=(256, 256))
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clip_max_norm", type=float, default=1.0)
    p.add_argument("--checkpoint", type=str, default=None)
    p.add_argument("--lr_end_threshold", type=float, default=2e-8,
                   help="LR early-stopping threshold (dtufc)")
    p.add_argument("--lr_patience", type=int, default=10,
                   help="Epochs to train after LR drops below threshold")
    p.add_argument("--eval_interval", type=int, default=20,
                   help="Downstream evaluation interval (epochs)")
    return p.parse_args(argv)


def main(argv):
    args = parse_args(argv)
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    is_seg = (args.task == "seg")

    centers = load_kmeans_centers(args.mapping)
    print(f"Loaded kmeans mapping: {args.mapping} ({len(centers)} centers)")
    print(f"Task: {args.task}")

    # --- datasets (train & test_crop are shared for both tasks) ---
    train_dataset = PreprocessedTrainFolder(args.train_data)
    test_crop_dataset = PreprocessedTestCropFolder(args.train_data)

    if is_seg:
        if not args.seg_test_root or not args.gt_root:
            raise ValueError("--seg_test_root and --gt_root are required for seg task")
        eval_dataset = PreprocessedSegTestFolder(
            quant_root=args.train_data,
            orig_root=args.seg_test_root,
            model_type=args.model_type,
            layer=args.layer,
            gt_root=args.gt_root,
            trun_low=args.trun_low,
            trun_high=args.trun_high,
        )
        print(f"Train: {len(train_dataset)}, Test crops: {len(test_crop_dataset)}, "
              f"Seg eval: {len(eval_dataset)}")
    else:
        if not args.test_data:
            raise ValueError("--test_data is required for cls task")
        eval_dataset = PreprocessedTestFolder(
            quant_root=args.train_data,
            orig_root=args.test_data,
            model_type=args.model_type,
            layer=args.layer,
            gt_path=args.gt_path,
            trun_low=args.trun_low,
            trun_high=args.trun_high,
        )
        print(f"Train: {len(train_dataset)}, Test crops: {len(test_crop_dataset)}, "
              f"Cls eval: {len(eval_dataset)}")

    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              num_workers=args.num_workers, shuffle=True,
                              pin_memory=(device == "cuda"))
    test_crop_loader = DataLoader(test_crop_dataset, batch_size=args.test_batch_size,
                                  num_workers=args.num_workers, shuffle=False,
                                  pin_memory=(device == "cuda"))

    eval_bs = 1 if is_seg else min(8, args.batch_size)
    eval_loader = DataLoader(eval_dataset, batch_size=eval_bs,
                             num_workers=min(args.num_workers, 8), shuffle=False,
                             pin_memory=(device == "cuda"))

    net = image_models[args.model](quality=1).to(device)

    layer_idx = int("".join(c for c in args.layer if c.isdigit()) or "0")
    dinowrapper = Dinov2Wrapper(head_layers=1, model_name=args.model_type, device=device)
    if is_seg:
        dinowrapper.load_segmentation_head(args.seg_head_path)

    conf = {
        "net": {"type": "Adam", "lr": args.learning_rate},
        "aux": {"type": "Adam", "lr": args.aux_learning_rate},
    }
    optim_dict = net_aux_optimizer(net, conf)
    optimizer = optim_dict["net"]
    aux_optimizer = optim_dict["aux"]
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")
    criterion = RateDistortionLoss(lmbda=args.lmbda)

    last_epoch = 0
    if args.checkpoint and args.checkpoint != 'None' and os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        last_epoch = ckpt["epoch"] + 1
        net.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        aux_optimizer.load_state_dict(ckpt["aux_optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])

    best_loss = float("inf")
    best_metric = -1.0
    writer = SummaryWriter(args.log_dir, purge_step=last_epoch)

    lr_below_threshold = False
    lr_patience_counter = 0

    for epoch in range(last_epoch, args.epochs):
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Learning rate: {current_lr}")

        train_one_epoch(net, criterion, train_loader, optimizer, aux_optimizer,
                        epoch, args.clip_max_norm, writer)

        loss = test_epoch(epoch, test_crop_loader, net, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        # Downstream evaluation (cls accuracy / seg mIoU)
        run_eval = (epoch % args.eval_interval == 0) or (epoch == args.epochs - 1)
        if run_eval:
            if is_seg:
                metric_val = seg_eval_epoch(epoch, eval_loader, net, dinowrapper,
                                            criterion, layer_idx, centers,
                                            args.bit_depth, writer)
            else:
                metric_val = cls_eval_epoch(epoch, eval_loader, net, dinowrapper,
                                            criterion, layer_idx, centers,
                                            args.bit_depth, writer)

            is_best_metric = metric_val > best_metric
            if is_best_metric:
                best_metric = metric_val

            if args.savepath:
                ckpt_state = {
                    "epoch": epoch, "state_dict": net.state_dict(), "loss": loss,
                    "metric": metric_val,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                }
                save_cls_checkpoint(ckpt_state, epoch, is_best_metric, args.savepath)

        gc.collect()
        torch.cuda.empty_cache()

        if args.savepath:
            if is_best:
                print(f'Current best epoch: {epoch}')
                save_checkpoint({
                    "epoch": epoch, "state_dict": net.state_dict(), "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                }, is_best=True, filename=args.savepath)

        # dtufc LR-based early stopping
        if current_lr <= args.lr_end_threshold:
            if not lr_below_threshold:
                print(f"LR dropped below {args.lr_end_threshold}, "
                      f"will train {args.lr_patience} more epochs.")
                lr_below_threshold = True
                lr_patience_counter = 0
            else:
                lr_patience_counter += 1
                print(f"LR below threshold for {lr_patience_counter}/{args.lr_patience} epochs.")
                if lr_patience_counter >= args.lr_patience:
                    if args.savepath:
                        save_checkpoint({
                            "epoch": epoch, "state_dict": net.state_dict(), "loss": loss,
                            "optimizer": optimizer.state_dict(),
                            "aux_optimizer": aux_optimizer.state_dict(),
                            "lr_scheduler": lr_scheduler.state_dict(),
                        }, is_best=False, filename=args.savepath)
                    print("Stopping training due to low learning rate.")
                    break

    if writer:
        writer.close()

    # Final evaluation with real entropy coding
    if args.savepath:
        cls_best_path = str(Path(args.savepath).parent / "checkpoint_cls_best.pth.tar")
        rd_best_path = get_best_path(args.savepath)
        eval_path = cls_best_path if os.path.exists(cls_best_path) else rd_best_path
        if os.path.exists(eval_path):
            print(f"\nLoading best checkpoint for final eval: {eval_path}")
            ckpt = torch.load(eval_path, map_location=device)
            model_cls = model_architectures[args.model]
            net_eval = model_cls.from_state_dict(ckpt["state_dict"]).to(device).eval()
            if is_seg:
                final_eval_seg(net_eval, eval_loader, dinowrapper, layer_idx,
                               centers, args.bit_depth, args.savepath)
            else:
                final_eval(net_eval, eval_loader, dinowrapper, layer_idx,
                           centers, args.bit_depth, args.savepath)
        else:
            print(f"No checkpoint found for final eval, skipping.")


if __name__ == "__main__":
    main(sys.argv[1:])

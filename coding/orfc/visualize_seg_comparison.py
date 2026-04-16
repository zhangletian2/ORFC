#!/usr/bin/env python
"""
Segmentation visualization comparison: VTM vs OPQ vs Ours.

For each method on the VOC2012 val-100 subset:
  1. Load / compute reconstructed features at blk20
  2. Run ViT tail (blk21-23 + norm) → segmentation head → slide inference
  3. Compute per-image mIoU
  4. Save boundary-overlay visualization with per-image mIoU annotation

Usage:
python visualize_seg_comparison.py \
  --gpu 0 \
  --layer blk10 \
  --K 8 \
  --vtm_qp 32 \
  --codec_path checkpoints/dinov2_vitl14/blk10_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt
"""

import os, sys, argparse, math, time
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

ORFC_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(ORFC_ROOT, "..", ".."))
if ORFC_ROOT not in sys.path:
    sys.path.insert(0, ORFC_ROOT)

from run_multilayer_calibrator import set_seed, preload_features
from opq import (
    batch_normalize_gpu, batch_inv_normalize_gpu,
    batched_assign, learn_opq_rotation,
)
from backbone.wrapper import (
    SegmentationEvaluator, load_seg_head, _DINOV2_REGISTRY,
)
from soft_pq import load_codec

import warnings
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
import logging
from mmcv.utils import get_logger
get_logger('mmcv').setLevel(logging.WARNING)


# ================================================================
#     VOC2012 palette & visualization helpers
# ================================================================

VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
NUM_CLASSES = 21
IGNORE_INDEX = 255


def voc_colormap(N=256):
    """Standard VOC2012 segmentation color palette."""
    cmap = np.zeros((N, 3), dtype=np.uint8)
    for i in range(N):
        r, g, b = 0, 0, 0
        c = i
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        cmap[i] = [r, g, b]
    return cmap


PALETTE = voc_colormap()


def get_boundary(seg, thickness=2, skip_background=True):
    """Extract boundary pixels where label transitions occur.

    If skip_background=True, only boundaries touching non-background
    classes are returned, reducing visual noise.
    """
    h, w = seg.shape
    boundary = np.zeros((h, w), dtype=bool)
    if h > 1:
        diff_v = seg[:-1, :] != seg[1:, :]
        boundary[:-1, :] |= diff_v
        boundary[1:, :] |= diff_v
    if w > 1:
        diff_h = seg[:, :-1] != seg[:, 1:]
        boundary[:, :-1] |= diff_h
        boundary[:, 1:] |= diff_h
    if skip_background:
        fg = seg > 0
        if thickness > 1:
            from scipy.ndimage import binary_dilation
            fg = binary_dilation(fg, iterations=thickness)
        boundary &= fg
    if thickness > 1:
        from scipy.ndimage import binary_dilation
        boundary = binary_dilation(boundary, iterations=thickness - 1)
    return boundary


def brighten_palette(palette, factor=1.4, floor=60):
    """Make palette colors brighter for boundary visibility."""
    bp = palette.astype(np.float32) * factor
    bp = np.clip(bp, floor, 255).astype(np.uint8)
    bp[0] = [0, 0, 0]  # keep background black
    return bp


BRIGHT_PALETTE = brighten_palette(PALETTE)


def overlay_boundary(img_rgb, seg_pred, thickness=2, alpha=0.95):
    """Overlay class-colored segmentation boundaries on the original image.

    Non-background class regions also get a light transparent fill.
    """
    canvas = img_rgb.copy().astype(np.float32)

    fg = seg_pred > 0
    if fg.any():
        fill_color = PALETTE[seg_pred].astype(np.float32)
        fg_3d = np.stack([fg] * 3, axis=-1)
        canvas = np.where(fg_3d,
                          0.75 * canvas + 0.25 * fill_color,
                          canvas)

    boundary = get_boundary(seg_pred, thickness=thickness, skip_background=True)
    if boundary.any():
        b_color = BRIGHT_PALETTE[seg_pred].astype(np.float32)
        b_3d = np.stack([boundary] * 3, axis=-1)
        canvas = np.where(b_3d,
                          (1 - alpha) * canvas + alpha * b_color,
                          canvas)

    return canvas.astype(np.uint8)


def overlay_mask_transparent(img_rgb, seg_pred, seg_gt=None, alpha=0.5):
    """Semi-transparent colored mask overlay (for GT panel)."""
    canvas = img_rgb.copy().astype(np.float32)
    mask_color = PALETTE[seg_pred].astype(np.float32)
    valid = seg_pred > 0
    if seg_gt is not None:
        valid &= (seg_gt != IGNORE_INDEX)
    valid_3d = np.stack([valid] * 3, axis=-1)
    canvas = np.where(valid_3d,
                      (1 - alpha) * canvas + alpha * mask_color,
                      canvas)
    boundary = get_boundary(seg_pred, thickness=3, skip_background=True)
    if boundary.any():
        b_color = BRIGHT_PALETTE[seg_pred].astype(np.float32)
        b_3d = np.stack([boundary] * 3, axis=-1)
        canvas = np.where(b_3d, b_color, canvas)
    return canvas.astype(np.uint8)


def compute_single_miou(pred, gt, num_classes=NUM_CLASSES):
    """Per-image mIoU from prediction and ground truth masks."""
    mask = gt != IGNORE_INDEX
    if mask.sum() == 0:
        return float('nan')
    hist = np.bincount(
        num_classes * gt[mask].astype(int) + pred[mask].astype(int),
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    return float(np.nanmean(iou))


# ================================================================
#     Normalization helpers (match SegmentationEvaluator)
# ================================================================

def per_image_norm(x, eps=1e-5):
    mu = x.mean()
    var = ((x - mu) ** 2).mean()
    std = np.sqrt(var + eps)
    return ((x - mu) / std).astype(np.float32), \
           np.array([[mu]], dtype=np.float32), \
           np.array([[std]], dtype=np.float32)


# ================================================================
#     Shared segmentation inference
# ================================================================

CROP_SIZE = (512, 512)
STRIDE = (341, 341)
PATCH_SIZE = 14


def get_slide_crops(h_img, w_img, crop_size=CROP_SIZE, stride=STRIDE):
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


@torch.no_grad()
def slide_inference(backbone, head, feature_list, crops, img_shape,
                    layer_idx, device, patch_size=PATCH_SIZE):
    """Slide decode + fusion → logits [1, C, H, W]."""
    h_img, w_img = img_shape
    h_crop, w_crop = CROP_SIZE
    preds = torch.zeros((1, NUM_CLASSES, h_img, w_img), device=device)
    count_mat = torch.zeros((1, 1, h_img, w_img), device=device)

    for i, (y1, x1, y2, x2) in enumerate(crops):
        feat = feature_list[i]
        if isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat).float().to(device)
        if feat.dim() == 2:
            feat = feat.unsqueeze(0)
        feat = feat.to(device)

        x = feat
        for blk_idx in range(layer_idx + 1, len(backbone.blocks)):
            x = backbone.blocks[blk_idx](x)
        x = backbone.norm(x)

        patch_tokens = x[:, 1:, :]
        actual_h = y2 - y1
        actual_w = x2 - x1
        padded_h = math.ceil(actual_h / patch_size) * patch_size
        padded_w = math.ceil(actual_w / patch_size) * patch_size
        feat_h = padded_h // patch_size
        feat_w = padded_w // patch_size
        patch_tokens = patch_tokens.reshape(1, feat_h, feat_w, -1).permute(0, 3, 1, 2)

        logits = head(patch_tokens)
        logits_up = F.interpolate(logits, size=(h_crop, w_crop),
                                  mode='bilinear', align_corners=False)
        logits_crop = logits_up[:, :, :actual_h, :actual_w]
        preds[:, :, y1:y2, x1:x2] += logits_crop
        count_mat[:, :, y1:y2, x1:x2] += 1

    preds = preds / count_mat.clamp(min=1)
    return preds


# ================================================================
#     Method-specific quantize_tokens
# ================================================================

@torch.no_grad()
def quantize_vtm(tokens_np, device):
    """VTM: features already decoded, just convert to tensor."""
    return torch.from_numpy(tokens_np.astype(np.float32)).to(device)


@torch.no_grad()
def quantize_opq(tokens_np, R_t, cb_t, num_groups, embedding_dim,
                 feat_dim, device):
    """OPQ: rotation + hard assign + inverse rotation."""
    y, mu, std = per_image_norm(tokens_np)
    Y = torch.from_numpy(y).float().to(device).unsqueeze(0)
    flat = Y.reshape(-1, feat_dim)
    Z = flat @ R_t
    z_3d = Z.reshape(-1, num_groups, embedding_dim).permute(1, 0, 2).contiguous()
    z_hat_3d, _ = batched_assign(z_3d, cb_t, device=device)
    flat_hat = z_hat_3d.permute(1, 0, 2).reshape(-1, feat_dim)
    Y_hat = (flat_hat @ R_t.T).reshape(1, -1, feat_dim)
    Mu = torch.from_numpy(mu).float().to(device).unsqueeze(0)
    Std = torch.from_numpy(std).float().to(device).unsqueeze(0)
    X_hat = Y_hat * Std + Mu
    return X_hat.squeeze(0)


@torch.no_grad()
def quantize_codec(tokens_np, codec, device):
    """Ours: codec encode/decode."""
    X = torch.from_numpy(tokens_np.astype(np.float32)).unsqueeze(0).to(device)
    Y, Mu, Std = batch_normalize_gpu(X, mode='per_image')
    Y_hat, _ = codec(Y)
    X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
    return X_hat.squeeze(0)


# ================================================================
#     OPQ training
# ================================================================

def train_opq(feat_dir, backbone_name, layer, K, embedding_dim,
              device, max_train=5000, seed=42):
    """Train standard OPQ on ImageNet features → (R, codebooks)."""
    feat_files = sorted(Path(feat_dir).glob("*.npy"))
    print(f"  Loading {len(feat_files)} training features ...")
    features, _ = preload_features(feat_files, num_workers=8)

    if max_train > 0 and len(features) > max_train:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(features), max_train, replace=False)
        features = [features[i] for i in idx]

    D = features[0].shape[1]
    G = D // embedding_dim

    all_vecs = []
    for s in range(0, len(features), 200):
        e = min(s + 200, len(features))
        X = torch.from_numpy(np.stack(features[s:e])).float().to(device)
        with torch.no_grad():
            Y, _, _ = batch_normalize_gpu(X, mode='per_image')
        all_vecs.append(Y.reshape(-1, D).cpu().numpy())
        del X, Y
    flat = np.concatenate(all_vecs, axis=0)
    del all_vecs

    max_flat = 2_000_000 // G
    if flat.shape[0] > max_flat:
        rng2 = np.random.RandomState(seed)
        flat = flat[rng2.choice(flat.shape[0], max_flat, replace=False)]

    print(f"  Training OPQ: G={G}, K={K}, d={embedding_dim}, "
          f"vectors={flat.shape[0]}")
    R, codebooks, hist = learn_opq_rotation(
        flat, G, embedding_dim, K,
        max_iter_opq=20, max_iter_kmeans=100,
        device=device, verbose=False)
    print(f"  OPQ trained. MSE={hist[-1][0]:.8f}")
    del flat
    torch.cuda.empty_cache()
    return R, codebooks


# ================================================================
#     Main
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Segmentation visualization: VTM vs OPQ vs Ours",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--layer", type=str, default="blk20")
    parser.add_argument("--K", type=int, default=8,
                        help="OPQ/Ours codebook size")
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--backbone", type=str, default="dinov2_vitl14")

    parser.add_argument("--feat_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "features"),
                        help="Root of all feature directories")
    parser.add_argument("--vtm_qp", type=int, default=32,
                        help="VTM QP value for decoded features")
    parser.add_argument("--vtm_feat_dir", type=str, default=None,
                        help="Override VTM feature dir (auto-derived if None)")
    parser.add_argument("--orig_feat_dir", type=str, default=None,
                        help="Override original feature dir (auto-derived if None)")
    parser.add_argument("--train_feat_dir", type=str, default=None,
                        help="Override training feature dir (auto-derived if None)")
    parser.add_argument("--codec_path", type=str, default=None,
                        help="Override codec checkpoint path (auto-derived if None)")

    parser.add_argument("--voc_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "data",
                                "VOCdevkit", "VOC2012"))
    parser.add_argument("--image_list", type=str,
                        default=os.path.join(PROJECT_ROOT, "utils",
                                "voc2012_val_100.txt"))
    parser.add_argument("--weights_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "pretrained"))
    parser.add_argument("--max_train", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--boundary_thickness", type=int, default=3)
    parser.add_argument("--font_size", type=int, default=10)
    parser.add_argument("--out_dir", type=str, default="")
    parser.add_argument("--select_images", nargs='*', default=None,
                        help="Only visualize these specific images")

    args = parser.parse_args()
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    set_seed(args.seed)

    layer_idx = int(args.layer[-2:])
    reg = _DINOV2_REGISTRY[args.backbone]
    D = reg["embed_dim"]

    # Auto-derive paths from --layer / --backbone / --K if not explicitly set
    if args.orig_feat_dir is None:
        args.orig_feat_dir = os.path.join(
            args.feat_root, "voc2012_100", args.backbone, args.layer)
    if args.vtm_feat_dir is None:
        args.vtm_feat_dir = os.path.join(
            args.feat_root, "voc2012_100", args.backbone,
            "decoded", "vtm", str(args.vtm_qp), args.layer)
    if args.train_feat_dir is None:
        args.train_feat_dir = os.path.join(
            args.feat_root, "train", args.backbone, args.layer)
    if args.codec_path is None:
        ckpt_dir = os.path.join(ORFC_ROOT, "checkpoints", args.backbone)
        pattern = f"{args.layer}_K{args.K}_emb{args.embedding_dim}_*_tau*_*.pt"
        import glob
        candidates = sorted(glob.glob(os.path.join(ckpt_dir, pattern)))
        if candidates:
            args.codec_path = candidates[-1]
        else:
            raise FileNotFoundError(
                f"No codec checkpoint found matching {pattern} in {ckpt_dir}")

    print(f"\n  Resolved paths for --layer {args.layer}:")
    print(f"    orig_feat_dir  = {args.orig_feat_dir}")
    print(f"    vtm_feat_dir   = {args.vtm_feat_dir}")
    print(f"    train_feat_dir = {args.train_feat_dir}")
    print(f"    codec_path     = {args.codec_path}")

    if not args.out_dir:
        args.out_dir = os.path.join(ORFC_ROOT, 'figures', 'seg_vis',
                                    f'{args.backbone}_{args.layer}_K{args.K}')
    comp_dir = os.path.join(args.out_dir, 'comparison')
    os.makedirs(comp_dir, exist_ok=True)

    with open(args.image_list) as f:
        val_list = [ln.strip() for ln in f if ln.strip()]
    if args.select_images:
        val_list = [n for n in val_list if n in args.select_images]
    print(f"\nImages to evaluate: {len(val_list)}")

    # ── Load backbone + seg head (shared across methods) ──
    print(f"\nLoading DINOv2 backbone ({args.backbone}) ...")
    from dinov2.models import vision_transformer as vits
    vit_builder = getattr(vits, reg["vit_fn"])
    backbone = vit_builder(**reg["vit_kwargs"])
    backbone_ckpt = os.path.join(args.weights_root, reg["pretrain"])
    backbone.load_state_dict(torch.load(backbone_ckpt, map_location="cpu"),
                             strict=True)
    backbone = backbone.to(device).eval()

    head_ckpt = os.path.join(args.weights_root, reg["seg_head"])
    seg_head = load_seg_head(head_ckpt, in_channels=D,
                             num_classes=NUM_CLASSES, device=device)
    print(f"  Backbone + seg head loaded.")

    # ── mmseg test pipeline for image preprocessing ──
    import mmcv
    from mmcv.parallel import collate
    from mmseg.datasets.pipelines import Compose

    cfg = mmcv.Config.fromfile(reg["config"])
    cfg.data_root = args.voc_root

    class _LoadImage:
        def __call__(self, results):
            results['filename'] = results['ori_filename'] = None
            img = results['img']
            results['img_shape'] = img.shape
            results['ori_shape'] = img.shape
            return results

    test_pipeline = Compose([_LoadImage()] + cfg.data.test.pipeline[1:])

    # ── Train OPQ ──
    print(f"\n{'=' * 60}")
    print(f"  Training OPQ: K={args.K}, emb={args.embedding_dim}")
    print(f"{'=' * 60}")
    R_opq, codebooks_opq = train_opq(
        args.train_feat_dir, args.backbone, args.layer,
        args.K, args.embedding_dim, device,
        max_train=args.max_train, seed=args.seed)
    num_groups = len(codebooks_opq)
    R_t = torch.from_numpy(R_opq).float().to(device)
    cb_t = torch.from_numpy(np.stack(codebooks_opq)).float().to(device)

    # ── Load our codec ──
    print(f"\n{'=' * 60}")
    print(f"  Loading trained codec: {args.codec_path}")
    print(f"{'=' * 60}")
    codec = load_codec(args.codec_path, device=device)
    codec.eval()
    print(f"  Codec loaded (G={codec.pq.G}, K={codec.pq.K}, d={codec.pq.d})")

    # ── Per-image evaluation & visualization ──
    print(f"\n{'=' * 60}")
    print(f"  Segmentation evaluation + visualization")
    print(f"{'=' * 60}")

    methods = ['original', 'vtm', 'opq', 'ours']
    method_labels = {
        'original': 'Original',
        'vtm': 'VTM',
        'opq': 'OPQ',
        'ours': 'Ours',
    }
    global_hist = {m: np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
                   for m in methods}
    per_image_miou = {m: {} for m in methods}

    for name in tqdm(val_list, desc="Seg evaluation"):
        img_path = os.path.join(args.voc_root, 'JPEGImages', f'{name}.jpg')
        gt_path = os.path.join(args.voc_root, 'SegmentationClass', f'{name}.png')
        orig_feat_path = os.path.join(args.orig_feat_dir, f'{name}.npy')
        vtm_feat_path = os.path.join(args.vtm_feat_dir, f'{name}.npy')

        if not os.path.exists(orig_feat_path):
            print(f"  [skip] {name}: original feature not found")
            continue

        img_pil = Image.open(img_path).convert('RGB')
        img_rgb = np.array(img_pil)
        img_bgr = img_rgb[:, :, ::-1]
        gt = np.array(Image.open(gt_path))
        ori_h, ori_w = gt.shape[:2]

        data = test_pipeline(dict(img=img_bgr.copy()))
        data = collate([data], samples_per_gpu=1)
        h_img = data['img'][0].shape[2]
        w_img = data['img'][0].shape[3]
        crops = get_slide_crops(h_img, w_img)

        orig_features = np.load(orig_feat_path)
        assert orig_features.shape[0] == len(crops), \
            f"{name}: slides mismatch {orig_features.shape[0]} vs {len(crops)}"

        vtm_features = None
        if os.path.exists(vtm_feat_path):
            vtm_features = np.load(vtm_feat_path)

        pred_masks = {}

        for method in methods:
            quant_list = []
            for s in range(orig_features.shape[0]):
                if method == 'original':
                    tokens = quantize_vtm(orig_features[s], device)
                elif method == 'vtm':
                    if vtm_features is None:
                        quant_list = None
                        break
                    tokens = quantize_vtm(vtm_features[s], device)
                elif method == 'opq':
                    tokens = quantize_opq(
                        orig_features[s], R_t, cb_t,
                        num_groups, args.embedding_dim, D, device)
                else:
                    tokens = quantize_codec(
                        orig_features[s], codec, device)
                quant_list.append(tokens.unsqueeze(0))

            if quant_list is None:
                continue

            preds_logits = slide_inference(
                backbone, seg_head, quant_list, crops,
                (h_img, w_img), layer_idx, device)

            if (h_img, w_img) != (ori_h, ori_w):
                preds_logits = F.interpolate(
                    preds_logits, size=(ori_h, ori_w),
                    mode='bilinear', align_corners=False)

            seg_pred = preds_logits.argmax(dim=1).squeeze(0).cpu().numpy()
            pred_masks[method] = seg_pred

            mask_valid = gt != IGNORE_INDEX
            global_hist[method] += np.bincount(
                NUM_CLASSES * gt[mask_valid].astype(int)
                + seg_pred[mask_valid].astype(int),
                minlength=NUM_CLASSES ** 2
            ).reshape(NUM_CLASSES, NUM_CLASSES)

            miou_img = compute_single_miou(seg_pred, gt)
            per_image_miou[method][name] = miou_img

            del quant_list, preds_logits
            torch.cuda.empty_cache()

        # ── Combined comparison figure (matplotlib for vector PDF) ──
        if len(pred_masks) == len(methods):
            gt_vis = overlay_mask_transparent(img_rgb, gt,
                                             seg_gt=gt, alpha=0.5)
            panels = [gt_vis]
            labels = ['GT']
            mious = [None]
            for m in methods:
                panels.append(overlay_boundary(
                    img_rgb, pred_masks[m],
                    thickness=args.boundary_thickness))
                labels.append(method_labels[m])
                mious.append(per_image_miou[m].get(name))

            n_panels = len(panels)
            panel_w = 2.4
            fig, axes = plt.subplots(
                1, n_panels,
                figsize=(panel_w * n_panels,
                         panel_w * ori_h / ori_w + 0.45))
            if n_panels == 1:
                axes = [axes]
            for pi, ax in enumerate(axes):
                ax.imshow(panels[pi])
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                miou_val = mious[pi]
                if miou_val is not None:
                    txt = f"{labels[pi]}  mIoU={miou_val:.2f}"
                else:
                    txt = labels[pi]
                ax.set_xlabel(txt, fontsize=args.font_size, fontweight='bold',
                              color='#1e1e1e', fontfamily='sans-serif',
                              labelpad=6)
            plt.subplots_adjust(wspace=0.02, left=0.01, right=0.99,
                                top=0.99, bottom=0.10)
            for ext in ['png', 'pdf']:
                fig.savefig(os.path.join(comp_dir, f'{name}.{ext}'),
                            dpi=300, bbox_inches='tight')
            plt.close(fig)

    # ── Global mIoU summary ──
    print(f"\n{'=' * 60}")
    print(f"  Global mIoU results")
    print(f"{'=' * 60}")
    for m in methods:
        h = global_hist[m]
        iou = np.diag(h) / (h.sum(1) + h.sum(0) - np.diag(h))
        g_miou = float(np.nanmean(iou))
        g_acc = float(np.diag(h).sum() / max(h.sum(), 1))
        n_imgs = len(per_image_miou[m])
        avg_per_img = float(np.nanmean(list(per_image_miou[m].values()))) \
            if n_imgs > 0 else 0.0
        print(f"  {method_labels[m]:12s}  "
              f"mIoU={g_miou:.4f}  aAcc={g_acc:.4f}  "
              f"avg_per_img_mIoU={avg_per_img:.4f}  "
              f"({n_imgs} images)")

    # ── Per-class IoU table ──
    print(f"\n  Per-class IoU:")
    header = f"  {'Class':>14s}"
    for m in methods:
        header += f"  {method_labels[m]:>12s}"
    print(header)
    for ci in range(NUM_CLASSES):
        row = f"  {VOC_CLASSES[ci]:>14s}"
        for m in methods:
            h = global_hist[m]
            denom = h[ci].sum() + h[:, ci].sum() - h[ci, ci]
            ciou = h[ci, ci] / denom if denom > 0 else float('nan')
            row += f"  {ciou:>12.4f}"
        print(row)

    print(f"\n  Visualizations saved to: {comp_dir}")
    print(f"  - PNG + PDF for each image")


if __name__ == '__main__':
    main()

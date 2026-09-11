#!/usr/bin/env python3
"""Replay ORFC Soft-PQ DINOv3 codecs on classification / ADE20K / NYUv2.

Classification: ImageNet 500 linear probe (probe trained on uncompressed 5k CLS).
Semseg: ADE20K val linear head (official DINOv3).
Depth: NYUv2 linear DPT head (official DINOv3).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

_EVAL_DIR = Path(__file__).resolve().parent
_ORFC = _EVAL_DIR.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))
if str(_ORFC) not in sys.path:
    sys.path.insert(0, str(_ORFC))
_ORFCV1 = _ORFC.parent / "orfcv1"
if str(_ORFCV1) not in sys.path:
    sys.path.insert(0, str(_ORFCV1))

from dinov3_eval_common import (  # noqa: E402
    ADE_ANN_DIR,
    ADE_FEAT_ROOT,
    BACKBONE_CKPT,
    CKPT_DIR,
    COFAI_ROOT,
    DEPTH_HEAD,
    EMBED_DIM,
    IMAGENET_TEST_FEAT,
    IMAGENET_TEST_LABELS,
    IMAGENET_TOKEN_HW,
    LAYERS,
    N_PREFIX,
    NORM_MODE,
    NYU_FEAT_ROOT,
    NYU_LIST,
    PATCH,
    PROBE_DIR,
    RESULTS_DIR,
    SEMSEG_HEAD,
    decode_slot,
    layer_idx,
    list_orfc_ckpts,
    load_label_map,
    nyu_stem_from_rel,
    parse_ckpt_name,
    resolve_nyu_depth,
)

os.environ.setdefault("PROJECT_ROOT", str(COFAI_ROOT))

from cofai.backbone.timm import Dinov3TimmBackbone  # noqa: E402
from cofai.heads import Dinov3DepthHead, Dinov3SegmentationHead  # noqa: E402
from cofai.metrics.cv_metrics import MeanIoUMetric, TopKAccuracyMetric  # noqa: E402
from cofai.metrics.depth_estimation import Dinov3DepthEstimationMeter  # noqa: E402
from cofai.transforms.core import PadToMultiple  # noqa: E402
from opq import batch_inv_normalize_gpu, batch_normalize_gpu  # noqa: E402
from soft_pq import load_codec, soft_pq_encode_decode  # noqa: E402
from absorb_r import AbsorbedFeatureCodec  # noqa: E402
from bilinear_residual import (  # noqa: E402
    BilinearORFCWrapper, BilinearSpatialCodec, freeze_module,
    load_residual_codec, load_spatial_weights, set_grid_hint,
)


# ---------------------------------------------------------------------------
# Backbone / codec helpers
# ---------------------------------------------------------------------------

def build_backbone(device, *, img_size=512, dynamic=True, slot=24):
    return (
        Dinov3TimmBackbone(
            model_size="large",
            img_size=img_size,
            patch_size=PATCH,
            dynamic_size=dynamic,
            slot=slot,
            n_last_blocks=1,
            cast_dtype="float32",
            pretrained=False,
            ckpt_path=BACKBONE_CKPT,
            device=str(device),
        )
        .eval()
        .to(device)
    )


def prime_rope(backbone, token_hw, device, cache: dict):
    key = (int(token_hw[0]), int(token_hw[1]))
    if key in cache:
        backbone._rope, backbone._attn_mask = cache[key]
        return
    h, w = key
    img = torch.zeros(1, 3, h * PATCH, w * PATCH, device=device)
    with torch.no_grad():
        backbone.encode(img)
    cache[key] = (backbone._rope, backbone._attn_mask)


@torch.no_grad()
def reconstruct_one(tokens: np.ndarray, codec, device,
                    token_hw=None) -> np.ndarray:
    if token_hw is not None:
        set_grid_hint(token_hw)
    x = torch.from_numpy(np.ascontiguousarray(tokens)).float().unsqueeze(0).to(device)
    y, mu, std = batch_normalize_gpu(x, mode=NORM_MODE, n_prefix=N_PREFIX)
    y_hat, _ = codec(y)
    x_hat = batch_inv_normalize_gpu(y_hat, mu, std)
    return x_hat[0].cpu().numpy()


def reconstruct_list(features, codec, device):
    if codec is None:
        return [np.asarray(f, dtype=np.float32) for f in features]
    return soft_pq_encode_decode(
        [np.asarray(f, dtype=np.float32) for f in features],
        codec,
        NORM_MODE,
        device,
        n_prefix=N_PREFIX,
    )


def load_npy_map(feat_dir: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(feat_dir.glob("*.npy"))}


def result_path(task: str, tag: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{task}_{tag}.json"


def dump_result(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)
    print(f"[ok] {path.name}: {payload.get('metrics')}")


def maybe_skip(path: Path, force: bool) -> bool:
    if path.is_file() and not force:
        print(f"[skip] exists {path}")
        return True
    return False


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def load_probe(layer: str, device):
    pt = PROBE_DIR / f"{layer}_linear_cls.pt"
    if not pt.is_file():
        raise FileNotFoundError(
            f"Missing linear probe {pt}. Run train_dinov3_cls_probe.py first."
        )
    blob = torch.load(pt, map_location="cpu")
    head = torch.nn.Linear(EMBED_DIM, 1000)
    head.weight.data.copy_(blob["weight"])
    head.bias.data.copy_(blob["bias"])
    return head.to(device).eval()


@torch.no_grad()
def eval_cls(args, backbone, codec, layer: str, device, tag: str):
    out = result_path("cls", tag)
    if maybe_skip(out, args.force):
        return
    feat_dir = IMAGENET_TEST_FEAT / layer
    labels = load_label_map(IMAGENET_TEST_LABELS)
    files = sorted(feat_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"no test features in {feat_dir}")
    feats, ys, names = [], [], []
    for p in files:
        y = labels.get(p.stem)
        if y is None:
            continue
        arr = np.load(p).astype(np.float32)
        if arr.ndim == 3:
            arr = arr.squeeze(0)
        feats.append(arr)
        ys.append(y)
        names.append(p.stem)
    recon = reconstruct_list(feats, codec, device)
    mse = float(np.mean([np.mean((a - b) ** 2) for a, b in zip(feats, recon)]))

    backbone.slot = decode_slot(layer)
    rope_cache = {}
    prime_rope(backbone, IMAGENET_TOKEN_HW, device, rope_cache)
    probe = load_probe(layer, device)
    meter = TopKAccuracyMetric(topk=[1, 5])
    bs = args.cls_batch_size
    for i in tqdm(range(0, len(recon), bs), desc=f"cls-{tag}"):
        chunk = recon[i : i + bs]
        x = torch.from_numpy(np.stack(chunk)).float().to(device)
        cls = backbone.decode_cls(x, IMAGENET_TOKEN_HW)[-1][0]
        logits = probe(cls)
        pred = torch.topk(logits, k=5, dim=-1).indices.cpu().tolist()
        meter.update(pred, ys[i : i + bs])
        del x, cls, logits
    metrics = meter.compute()
    payload = {
        "task": "cls",
        "tag": tag,
        "layer": layer,
        "mode": "bypass" if codec is None else "orfc",
        "ckpt": args.ckpt_path,
        "n_samples": len(ys),
        "avg_mse": mse,
        "metrics": {k: float(v) for k, v in metrics.items()},
        "norm_mode": NORM_MODE,
    }
    dump_result(out, payload)


# ---------------------------------------------------------------------------
# Semseg
# ---------------------------------------------------------------------------

def load_ade_gt_padded(stem: str) -> np.ndarray:
    png = ADE_ANN_DIR / f"{stem}.png"
    seg = np.array(Image.open(png)).astype(np.int64) - 1
    seg[seg == -1] = 255
    sample = {"semseg": np.expand_dims(seg.astype(np.float32), -1)}
    sample = PadToMultiple(PATCH, keys=["semseg"])(sample)
    return sample["semseg"][..., 0]


def build_seg_head(device):
    return (
        Dinov3SegmentationHead(
            in_channels=[EMBED_DIM],
            in_index=[0],
            input_transform="resize_concat",
            channels=EMBED_DIM,
            num_classes=150,
            patch_size=PATCH,
            checkpoint=SEMSEG_HEAD,
        )
        .to(device)
        .eval()
    )


@torch.no_grad()
def eval_semseg(args, backbone, codec, layer: str, device, tag: str):
    out = result_path("semseg", tag)
    if maybe_skip(out, args.force):
        return
    feat_dir = ADE_FEAT_ROOT / layer
    meta_dir = ADE_FEAT_ROOT / "meta"
    files = sorted(feat_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"no ADE features in {feat_dir}")
    seg_n = args.seg_max_images if args.seg_max_images > 0 else args.max_images
    if seg_n > 0:
        files = files[:seg_n]      # sorted() -> deterministic fixed subset
        print(f"  [semseg] fixed subset: {len(files)} images")

    backbone.slot = decode_slot(layer)
    head = build_seg_head(device)
    meter = MeanIoUMetric(num_classes=150)
    rope_cache = {}
    total_mse = 0.0
    n = 0
    for p in tqdm(files, desc=f"semseg-{tag}"):
        stem = p.stem
        meta = dict(np.load(meta_dir / f"{stem}.npz"))
        token_hw = tuple(int(x) for x in meta["token_hw"])
        tokens = np.load(p).astype(np.float32)
        if tokens.ndim == 3:
            tokens = tokens.squeeze(0)
        recon = reconstruct_one(tokens, codec, device,
                                token_hw=token_hw) if codec is not None else tokens
        total_mse += float(np.mean((tokens - recon) ** 2))
        n += 1
        prime_rope(backbone, token_hw, device, rope_cache)
        h = torch.from_numpy(recon).unsqueeze(0).to(device)
        feat = backbone.decode_seg(h, token_hw)
        logits = head.predict(feat, scale=int(PATCH), token_hw=token_hw)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
        gt = load_ade_gt_padded(stem)
        if pred.shape != gt.shape:
            # crop to overlap if pad mismatch
            hmin = min(pred.shape[0], gt.shape[0])
            wmin = min(pred.shape[1], gt.shape[1])
            pred = pred[:hmin, :wmin]
            gt = gt[:hmin, :wmin]
        meter.update(pred, gt)
        del h, feat, logits
    metrics = meter.compute()
    payload = {
        "task": "semseg",
        "tag": tag,
        "layer": layer,
        "mode": "bypass" if codec is None else "orfc",
        "ckpt": args.ckpt_path,
        "n_samples": n,
        "avg_mse": total_mse / max(n, 1),
        "metrics": {"mIoU": float(metrics["mIoU"])},
        "norm_mode": NORM_MODE,
    }
    dump_result(out, payload)
    del head
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Depth
# ---------------------------------------------------------------------------

def load_nyu_depth_map():
    out = {}
    with open(NYU_LIST) as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) < 2:
                continue
            stem = nyu_stem_from_rel(parts[0])
            out[stem] = resolve_nyu_depth(parts[1])
    return out


def build_depth_head(device):
    return (
        Dinov3DepthHead(
            in_channels=[EMBED_DIM],
            min_depth=0.001,
            max_depth=10.0,
            n_output_channels=256,
            use_backbone_norm=True,
            use_batchnorm=True,
            use_cls_token=False,
            bins_strategy="linear",
            norm_strategy="linear",
            checkpoint=DEPTH_HEAD,
        )
        .to(device)
        .eval()
    )


@torch.no_grad()
def eval_depth(args, backbone, codec, layer: str, device, tag: str):
    out = result_path("depth", tag)
    if maybe_skip(out, args.force):
        return
    feat_dir = NYU_FEAT_ROOT / layer
    meta_dir = NYU_FEAT_ROOT / "meta"
    files = sorted(feat_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"no NYU features in {feat_dir}")
    if args.nyu_split_file:
        keep = set()
        with open(args.nyu_split_file) as fh:
            for line in fh:
                parts = line.split()
                if parts:
                    keep.add(nyu_stem_from_rel(parts[0]))
        files = [f for f in files if f.stem in keep]
        if not files:
            raise FileNotFoundError(
                f"no NYU features matched {args.nyu_split_file}")
        print(f"  [depth] split filter {args.nyu_split_file}: {len(files)} images")
    if args.max_images > 0:
        files = files[: args.max_images]
    depth_map = load_nyu_depth_map()

    backbone.slot = decode_slot(layer)
    head = build_depth_head(device)
    meter = Dinov3DepthEstimationMeter(
        names=["rmse", "abs_rel", "a1"],
        min_depth=0.001,
        max_depth=10.0,
        ignored_value=0.0,
        eval_mask="NYU_EIGEN",
        normalization_constant=1000.0,
    )
    rope_cache = {}
    total_mse = 0.0
    n = 0
    missing = 0
    for p in tqdm(files, desc=f"depth-{tag}"):
        stem = p.stem
        gt_path = depth_map.get(stem)
        if gt_path is None or not Path(gt_path).is_file():
            missing += 1
            continue
        meta = dict(np.load(meta_dir / f"{stem}.npz"))
        token_hw = tuple(int(x) for x in meta["token_hw"])
        img_hw = tuple(int(x) for x in meta["img_hw"])
        tokens = np.load(p).astype(np.float32)
        if tokens.ndim == 3:
            tokens = tokens.squeeze(0)
        recon = reconstruct_one(tokens, codec, device,
                                token_hw=token_hw) if codec is not None else tokens
        total_mse += float(np.mean((tokens - recon) ** 2))
        n += 1
        prime_rope(backbone, token_hw, device, rope_cache)
        h = torch.from_numpy(recon).unsqueeze(0).to(device)
        feat = backbone.decode_depth(h, token_hw)
        pred = head.predict(feat, size=(img_hw[0], img_hw[1]))
        gt = np.array(Image.open(gt_path), dtype=np.float32)
        meter.update(pred, gt)
        del h, feat, pred
    metrics = meter.compute()
    payload = {
        "task": "depth",
        "tag": tag,
        "layer": layer,
        "mode": "bypass" if codec is None else "orfc",
        "ckpt": args.ckpt_path,
        "n_samples": n,
        "missing_gt": missing,
        "avg_mse": total_mse / max(n, 1),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "norm_mode": NORM_MODE,
    }
    dump_result(out, payload)
    del head
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser("ORFC DINOv3 task eval")
    p.add_argument("--ckpt_path", default="", help="Soft-PQ .pt; empty with --bypass")
    p.add_argument("--bypass", action="store_true", help="Uncompressed feature replay")
    p.add_argument("--layer", default="", help="Required for --bypass; else parsed from ckpt")
    p.add_argument("--tasks", default="cls,semseg,depth")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--force", action="store_true")
    p.add_argument("--max_images", type=int, default=0, help="debug subset; 0=all")
    p.add_argument("--seg_max_images", type=int, default=0,
                   help="ADE20K subset size (first N of sorted list); 0=use --max_images")
    p.add_argument("--cls_batch_size", type=int, default=32)
    p.add_argument("--ckpt_dir", default=str(CKPT_DIR))
    p.add_argument("--stage1_ckpt", default="",
                   help="orfcv1 stage-1 bilinear codec .pt (spatial+residual, no PQ)")
    p.add_argument("--stage1_ablation", default="main",
                   choices=("main", "recon0", "full"))
    p.add_argument("--orfc_ckpt", default="",
                   help="stage-2 PQ codec .pt; with --stage1_ckpt pointing at "
                        "the matching *_spatial.pt this evaluates the joint "
                        "stage-2 cascade (spatial -> ORFC -> residual). Pass "
                        "the *_absorbed.pt / *_absorbed_spatial.pt pair to "
                        "evaluate the R-absorbed model instead.")
    p.add_argument("--nyu_split_file", default="",
                   help="restrict depth eval to images listed here (e.g. nyu_test_80.txt)")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    codec = None
    if args.stage1_ckpt:
        ckpt = Path(args.stage1_ckpt)
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        layer = args.layer
        if layer not in LAYERS:
            raise SystemExit(f"--layer must be one of {LAYERS}")
        residual, meta = load_residual_codec(str(ckpt), device=device)
        spatial = BilinearSpatialCodec(
            int(meta["D"]), n_prefix=N_PREFIX, scale=2,
            down=meta.get("spatial_down", "conv2"),
            up=meta.get("spatial_up", "conv2"),
            cls_mode=meta.get("cls_mode", "conv2")).to(device)
        load_spatial_weights(spatial, meta)
        orfc = None
        absorbed = False
        if args.orfc_ckpt:
            if not Path(args.orfc_ckpt).is_file():
                raise FileNotFoundError(args.orfc_ckpt)
            orfc = freeze_module(load_codec(args.orfc_ckpt, device=device))
            # An absorbed checkpoint (written by orfcv1/absorb_r.py) has no
            # transform left: R lives folded in the conv weights for the patch
            # tokens and as a dense matrix for the prefix, which bypasses the
            # conv.  Detect it by the stored R_dense and re-attach that half.
            _om = torch.load(args.orfc_ckpt, map_location="cpu")
            _R = _om.get("R_dense", None)
            if _R is not None:
                orfc = freeze_module(AbsorbedFeatureCodec(
                    orfc, _R.float().to(device),
                    n_prefix=N_PREFIX).to(device).eval())
                absorbed = True
        codec = BilinearORFCWrapper(
            freeze_module(spatial), orfc=orfc,
            residual=freeze_module(residual), n_prefix=N_PREFIX,
            residual_quantize=False,
            residual_ablation=args.stage1_ablation).to(device)
        codec.eval()
        stage = "stage1" if orfc is None else ("absorbed" if absorbed else "stage2")
        base = Path(args.orfc_ckpt).name[:-3] if orfc is not None else ckpt.name[:-3]
        tag = f"{stage}_{args.stage1_ablation}_{base}"
        print(f"[eval] {stage} codec spatial={ckpt}")
        if orfc is not None:
            print(f"[eval] {stage} orfc={args.orfc_ckpt}")
        print(f"[eval] layer={layer} ablation={args.stage1_ablation} "
              f"n_prefix={N_PREFIX} norm={NORM_MODE}")
        print(f"[eval] tag={tag}")
    elif args.bypass:
        layer = args.layer
        if layer not in LAYERS:
            raise SystemExit(f"--layer must be one of {LAYERS}")
        tag = f"bypass_{layer}"
        print(f"[eval] BYPASS layer={layer} tasks={tasks}")
    else:
        if not args.ckpt_path:
            raise SystemExit("provide --ckpt_path or --bypass")
        ckpt = Path(args.ckpt_path)
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        info = parse_ckpt_name(ckpt)
        layer = args.layer or info["layer"]
        if layer not in LAYERS:
            raise SystemExit(f"cannot parse layer from {ckpt.name}")
        print(f"[eval] load codec {ckpt}")
        codec = load_codec(str(ckpt), device=device)
        codec.eval()
        tag = ckpt.stem if not ckpt.name.endswith(".pt") else ckpt.name[:-3]
        # Path.stem strips only last suffix, good for lmbda0.0_...
        tag = ckpt.name[:-3] if ckpt.suffix == ".pt" else ckpt.stem
        print(f"[eval] layer={layer} K={info.get('K')} lmbda={info.get('lmbda')} tag={tag}")

    need_dynamic = any(t in tasks for t in ("semseg", "depth"))
    backbone = build_backbone(
        device,
        img_size=224 if tasks == ["cls"] else 512,
        dynamic=need_dynamic or True,
        slot=decode_slot(layer),
    )
    t0 = time.time()
    if "cls" in tasks:
        eval_cls(args, backbone, codec, layer, device, tag)
    if "semseg" in tasks:
        eval_semseg(args, backbone, codec, layer, device, tag)
    if "depth" in tasks:
        eval_depth(args, backbone, codec, layer, device, tag)
    print(f"[eval] done {tag} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

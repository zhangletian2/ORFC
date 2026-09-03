#!/usr/bin/env python3
"""Train a frozen DINOv3 linear ImageNet probe on uncompressed 5k train CLS.

For each layer: replay train tokens through remaining ViT blocks + LN, take CLS,
fit sklearn LogisticRegression (1000-way). Used as the classification head for
ORFC codec replay (same protocol as official ADE/NYU linear heads).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

_EVAL_DIR = Path(__file__).resolve().parent
_ORFC = _EVAL_DIR.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))
if str(_ORFC) not in sys.path:
    sys.path.insert(0, str(_ORFC))

from dinov3_eval_common import (  # noqa: E402
    BACKBONE_CKPT,
    COFAI_ROOT,
    IMAGENET_TOKEN_HW,
    IMAGENET_TRAIN_FEAT,
    IMAGENET_TRAIN_LABELS,
    LAYERS,
    PROBE_DIR,
    decode_slot,
    layer_idx,
    load_label_map,
)

os.environ.setdefault("PROJECT_ROOT", str(COFAI_ROOT))

from cofai.backbone.timm import Dinov3TimmBackbone  # noqa: E402


def prime_rope(backbone, token_hw, device):
    h, w = int(token_hw[0]), int(token_hw[1])
    img = torch.zeros(1, 3, h * 16, w * 16, device=device)
    with torch.no_grad():
        backbone.encode(img)


def build_backbone(device):
    bb = Dinov3TimmBackbone(
        model_size="large",
        img_size=224,
        patch_size=16,
        dynamic_size=False,
        slot=24,
        n_last_blocks=1,
        pretrained=False,
        ckpt_path=BACKBONE_CKPT,
        device=str(device),
    ).eval().to(device)
    return bb


@torch.no_grad()
def tokens_to_cls(backbone, tokens: torch.Tensor, token_hw):
    """tokens [B,T,D] -> CLS [B,D] after remaining blocks + final LN."""
    out = backbone.decode_cls(tokens, token_hw)
    return out[-1][0]


def load_layer_features(feat_dir: Path, labels: dict[str, int]):
    files = sorted(feat_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"no features in {feat_dir}")
    xs, ys, names = [], [], []
    missing = 0
    for p in files:
        y = labels.get(p.stem)
        if y is None:
            missing += 1
            continue
        arr = np.load(p).astype(np.float32)
        if arr.ndim == 3:
            arr = arr.squeeze(0)
        xs.append(arr)
        ys.append(y)
        names.append(p.stem)
    if missing:
        print(f"  [warn] {missing} npy without labels")
    return xs, np.asarray(ys, dtype=np.int64), names


@torch.no_grad()
def extract_cls_bank(backbone, features, device, batch_size=32):
    prime_rope(backbone, IMAGENET_TOKEN_HW, device)
    cls_all = []
    for i in tqdm(range(0, len(features), batch_size), desc="cls-bank"):
        chunk = features[i : i + batch_size]
        x = torch.from_numpy(np.stack(chunk)).float().to(device)
        cls = tokens_to_cls(backbone, x, IMAGENET_TOKEN_HW)
        cls_all.append(cls.float().cpu().numpy())
        del x, cls
    return np.concatenate(cls_all, axis=0)


def train_one_layer(layer: str, device, args):
    feat_dir = Path(args.train_feat) / layer
    labels = load_label_map(Path(args.label_file))
    print(f"\n=== {layer}  feat={feat_dir} ===")
    feats, y, names = load_layer_features(feat_dir, labels)
    print(f"  N={len(feats)}  classes={len(set(y.tolist()))}")

    backbone = build_backbone(device)
    backbone.slot = decode_slot(layer)
    print(f"  decode slot={backbone.slot} (layer_idx={layer_idx(layer)})")
    X = extract_cls_bank(backbone, feats, device, batch_size=args.batch_size)
    del backbone
    torch.cuda.empty_cache()

    print(f"  fitting LogisticRegression C={args.C} ...")
    clf = LogisticRegression(
        C=args.C,
        max_iter=args.max_iter,
        solver="lbfgs",
        verbose=0,
    )
    clf.fit(X, y)
    train_acc = float((clf.predict(X) == y).mean() * 100.0)
    print(f"  train acc={train_acc:.2f}%  n_classes={len(clf.classes_)}")

    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    out_pt = PROBE_DIR / f"{layer}_linear_cls.pt"
    weight = np.zeros((1000, X.shape[1]), dtype=np.float32)
    bias = np.zeros((1000,), dtype=np.float32)
    for i, c in enumerate(clf.classes_):
        weight[int(c)] = clf.coef_[i]
        bias[int(c)] = clf.intercept_[i]
    payload = {
        "weight": torch.from_numpy(weight),
        "bias": torch.from_numpy(bias),
        "classes": torch.from_numpy(clf.classes_.astype(np.int64)),
        "layer": layer,
        "C": args.C,
        "train_acc": train_acc,
        "n_train": int(len(y)),
        "names": names,
    }
    torch.save(payload, out_pt)
    meta = {
        "layer": layer,
        "C": args.C,
        "train_acc": train_acc,
        "n_train": int(len(y)),
        "n_classes": int(len(clf.classes_)),
        "path": str(out_pt),
    }
    with open(PROBE_DIR / f"{layer}_linear_cls.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  saved {out_pt}")
    return meta


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", nargs="+", default=list(LAYERS))
    p.add_argument("--train_feat", default=str(IMAGENET_TRAIN_FEAT))
    p.add_argument("--label_file", default=str(IMAGENET_TRAIN_LABELS))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--max_iter", type=int, default=2000)
    p.add_argument("--n_jobs", type=int, default=8)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows = []
    for layer in args.layers:
        rows.append(train_one_layer(layer, device, args))
    print("\n==== probe summary ====")
    for r in rows:
        print(f"  {r['layer']}: train_acc={r['train_acc']:.2f}%  n={r['n_train']}")


if __name__ == "__main__":
    main()

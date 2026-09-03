#!/usr/bin/env python3
"""Extract DINOv3 ViT-L/16 intermediate tokens for ORFC task eval.

Datasets:
  imagenet — 500-image test list, Resize256+CenterCrop224 (T=201)
  ade      — ADE20K val, CoFAI center PadToMultiple(16), variable T
  nyu      — NYUv2 test list, same center pad

All four layers (blk05/10/15/20) from one forward via block hooks.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

_EVAL_DIR = Path(__file__).resolve().parent
_ORFC = _EVAL_DIR.parent
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))
if str(_ORFC) not in sys.path:
    sys.path.insert(0, str(_ORFC))

from dinov3_eval_common import (  # noqa: E402
    ADE_ANN_DIR,
    ADE_FEAT_ROOT,
    ADE_IMG_DIR,
    BACKBONE_CKPT,
    COFAI_ROOT,
    IMAGENET_ROOT,
    IMAGENET_T,
    IMAGENET_TEST_FEAT,
    IMAGENET_TEST_LIST,
    LAYERS,
    NYU_FEAT_ROOT,
    NYU_LIST,
    PATCH,
    nyu_stem_from_rel,
    resolve_nyu_depth,
    resolve_nyu_rgb,
)

os.environ.setdefault("PROJECT_ROOT", str(COFAI_ROOT))

from cofai.backbone.timm import Dinov3TimmBackbone  # noqa: E402
from cofai.transforms.core import PadToMultiple  # noqa: E402


class BlockOutputCatcher:
    def __init__(self, backbone: nn.Module, block_indices):
        self.indices = sorted(set(int(i) for i in block_indices))
        self._buf = {}
        self._handles = []
        blocks = list(backbone.blocks)

        def _make_hook(idx):
            key = f"blk{idx:02d}"

            def hook(module, inp, out):
                t = out[0] if isinstance(out, tuple) else out
                self._buf[key] = t.detach()

            return hook

        for idx in self.indices:
            if idx >= len(blocks):
                raise IndexError(f"block {idx} out of range ({len(blocks)} blocks)")
            self._handles.append(blocks[idx].register_forward_hook(_make_hook(idx)))

    def pop(self):
        outs, self._buf = self._buf, {}
        return outs

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


def build_imagenet_tfm():
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )


def load_imagenet_pairs(list_txt: Path, root: Path):
    pairs = []
    with open(list_txt) as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) < 2:
                continue
            wnid, base = parts[0], parts[1]
            pairs.append((base, root / wnid / f"{base}.JPEG"))
    return pairs


def load_ade_items():
    items = []
    for p in sorted(ADE_IMG_DIR.glob("*.jpg")):
        items.append((p.stem, p, ADE_ANN_DIR / f"{p.stem}.png"))
    return items


def load_nyu_items():
    items = []
    with open(NYU_LIST) as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) < 2:
                continue
            img_rel, depth_rel = parts[0], parts[1]
            stem = nyu_stem_from_rel(img_rel)
            items.append((stem, resolve_nyu_rgb(img_rel), resolve_nyu_depth(depth_rel)))
    return items


def pad_hwc(img_hwc: np.ndarray) -> np.ndarray:
    padder = PadToMultiple(PATCH, keys=["img"])
    return padder({"img": img_hwc.astype(np.float32)})["img"]


def hwc_to_nchw(img_hwc: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.transpose(img_hwc, (2, 0, 1))).float()


def save_meta(meta_dir: Path, stem: str, token_hw, img_hw, padded_hw):
    meta_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        meta_dir / f"{stem}.npz",
        token_hw=np.array(token_hw, dtype=np.int32),
        img_hw=np.array(img_hw, dtype=np.int32),
        padded_hw=np.array(padded_hw, dtype=np.int32),
    )


def build_backbone(device, dynamic: bool):
    backbone = Dinov3TimmBackbone(
        model_size="large",
        img_size=224 if not dynamic else 512,
        patch_size=PATCH,
        dynamic_size=dynamic,
        slot=24,
        n_last_blocks=1,
        pretrained=False,
        ckpt_path=BACKBONE_CKPT,
        device=str(device),
    )
    backbone.model.to(device).eval()
    return backbone


@torch.no_grad()
def extract_imagenet(args, device):
    layers = [int(x.replace("blk", "")) for x in args.layers]
    out_root = Path(args.imagenet_out)
    for k in layers:
        (out_root / f"blk{k:02d}").mkdir(parents=True, exist_ok=True)

    pairs = load_imagenet_pairs(Path(args.imagenet_list), Path(args.imagenet_root))
    if args.max_images > 0:
        pairs = pairs[: args.max_images]
    print(f"[imagenet] N={len(pairs)}  out={out_root}  layers={args.layers}")

    backbone = build_backbone(device, dynamic=False)
    catcher = BlockOutputCatcher(backbone.model, layers)
    tfm = build_imagenet_tfm()
    saved = skipped = missing = 0
    t0 = time.time()
    try:
        for start in tqdm(range(0, len(pairs), args.batch_size), desc="imagenet"):
            batch = pairs[start : start + args.batch_size]
            todo = []
            for base, path in batch:
                need = False
                for k in layers:
                    fp = out_root / f"blk{k:02d}" / f"{base}.npy"
                    if not (args.skip_existing and fp.is_file()):
                        need = True
                        break
                if not need:
                    skipped += 1
                    continue
                if not path.is_file():
                    print(f"[warn] missing {path}")
                    missing += 1
                    continue
                todo.append((base, path))
            if not todo:
                continue
            imgs = [tfm(Image.open(p).convert("RGB")) for _, p in todo]
            x = torch.stack(imgs).to(device)
            _ = backbone.encode(x)
            outs = catcher.pop()
            for k in layers:
                arr = outs[f"blk{k:02d}"].float().cpu().numpy()
                layer_dir = out_root / f"blk{k:02d}"
                for i, (base, _) in enumerate(todo):
                    rec = arr[i]
                    if rec.shape != (IMAGENET_T, 1024):
                        raise SystemExit(f"imagenet shape {rec.shape} != ({IMAGENET_T}, 1024)")
                    np.save(layer_dir / f"{base}.npy", rec.astype(np.float32))
            saved += len(todo)
            del x, outs
    finally:
        catcher.close()
    print(
        f"[imagenet] saved={saved} skipped={skipped} missing={missing}  "
        f"{time.time() - t0:.1f}s"
    )


def _variable_need(out_root: Path, stem: str, layers, skip: bool) -> bool:
    if not skip:
        return True
    meta = out_root / "meta" / f"{stem}.npz"
    if not meta.is_file():
        return True
    for k in layers:
        if not (out_root / f"blk{k:02d}" / f"{stem}.npy").is_file():
            return True
    return False


@torch.no_grad()
def extract_variable(args, device, *, name: str, items, out_root: Path):
    """ADE / NYU: center-pad, group by padded HxW, hook all layers."""
    layers = [int(x.replace("blk", "")) for x in args.layers]
    out_root = Path(out_root)
    meta_dir = out_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    for k in layers:
        (out_root / f"blk{k:02d}").mkdir(parents=True, exist_ok=True)

    if args.max_images > 0:
        items = items[: args.max_images]
    print(f"[{name}] N={len(items)}  out={out_root}  layers={args.layers}")

    backbone = build_backbone(device, dynamic=True)
    catcher = BlockOutputCatcher(backbone.model, layers)

    groups = defaultdict(list)
    skipped = 0
    for stem, img_path, _gt in items:
        if _variable_need(out_root, stem, layers, args.skip_existing):
            im = Image.open(img_path).convert("RGB")
            arr = np.asarray(im).astype(np.float32) / 255.0
            padded = pad_hwc(arr)
            ph, pw = int(padded.shape[0]), int(padded.shape[1])
            groups[(ph, pw)].append(
                (stem, padded, (int(arr.shape[0]), int(arr.shape[1])))
            )
        else:
            skipped += 1
    print(f"[{name}] unique padded sizes={len(groups)}  skip={skipped}  todo={sum(len(v) for v in groups.values())}")

    saved = 0
    t0 = time.time()
    try:
        for (ph, pw), recs in tqdm(sorted(groups.items()), desc=f"{name}-sizes"):
            token_hw = (ph // PATCH, pw // PATCH)
            expect_t = N_PREFIX_PLUS(token_hw)
            for start in range(0, len(recs), args.batch_size):
                chunk = recs[start : start + args.batch_size]
                x = torch.stack([hwc_to_nchw(p) for _, p, _ in chunk]).to(device)
                _ = backbone.encode(x)
                outs = catcher.pop()
                for stem, _padded, img_hw in chunk:
                    save_meta(meta_dir, stem, token_hw, img_hw, (ph, pw))
                for k in layers:
                    arr = outs[f"blk{k:02d}"].float().cpu().numpy()
                    layer_dir = out_root / f"blk{k:02d}"
                    for i, (stem, _, _) in enumerate(chunk):
                        rec = arr[i]
                        if rec.shape != (expect_t, 1024):
                            raise SystemExit(
                                f"{name} {stem} {f'blk{k:02d}'} shape {rec.shape} "
                                f"!= ({expect_t}, 1024) token_hw={token_hw}"
                            )
                        np.save(layer_dir / f"{stem}.npy", rec.astype(np.float32))
                saved += len(chunk)
                del x, outs
    finally:
        catcher.close()
    print(f"[{name}] saved={saved} skipped={skipped}  {time.time() - t0:.1f}s")
    n_meta = len(list(meta_dir.glob("*.npz")))
    print(f"[{name}] meta={n_meta}")
    for k in layers:
        n = len(list((out_root / f"blk{k:02d}").glob("*.npy")))
        print(f"  blk{k:02d}: {n} npy")


def N_PREFIX_PLUS(token_hw):
    return 5 + int(token_hw[0]) * int(token_hw[1])


def parse_args():
    p = argparse.ArgumentParser("DINOv3 eval-set feature extract")
    p.add_argument("--dataset", choices=("imagenet", "ade", "nyu", "all"), default="all")
    p.add_argument("--layers", nargs="+", default=list(LAYERS))
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_images", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--skip_existing", action="store_true", default=True)
    p.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    p.add_argument("--imagenet_list", default=str(IMAGENET_TEST_LIST))
    p.add_argument("--imagenet_root", default=str(IMAGENET_ROOT))
    p.add_argument("--imagenet_out", default=str(IMAGENET_TEST_FEAT))
    p.add_argument("--ade_out", default=str(ADE_FEAT_ROOT))
    p.add_argument("--nyu_out", default=str(NYU_FEAT_ROOT))
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device={device}  datasets={args.dataset}  layers={args.layers}")
    ds = ("imagenet", "ade", "nyu") if args.dataset == "all" else (args.dataset,)
    if "imagenet" in ds:
        extract_imagenet(args, device)
    if "ade" in ds:
        extract_variable(
            args, device, name="ade", items=load_ade_items(), out_root=Path(args.ade_out)
        )
    if "nyu" in ds:
        extract_variable(
            args, device, name="nyu", items=load_nyu_items(), out_root=Path(args.nyu_out)
        )


if __name__ == "__main__":
    main()

#!/data4/workspace/zlt/featcodec/CoFAI/.venv/bin/python
"""
DINOv3 ViT-L/16 特征提取 — COCO train2017 可变分辨率版本

从 COCO train2017 随机选取指定数量的图片，使用 shorter-side=1024 (max 1536)
的 resize 策略提取指定 block 层特征，保存为 .npy (float32)。

每张图片保存 [T, 1024] 的完整 token 序列 (CLS + 4 reg + H×W patches)。

用法：
  python dinov3_coco_feat_extract.py \
      --coco_root /data4/workspace/zlt/compression_vit/data/coco \
      --out_root /data4/workspace/zlt/featcodec/features/train/dinov3_vitl16_coco/blk05 \
      --layer 5 --n_images 5000 --seed 42 --device cuda:0

  # 最后一层 (blk23, ViT-L/16 depth=24)
  python dinov3_coco_feat_extract.py \
      --out_root /data4/workspace/zlt/featcodec/features/train/dinov3_vitl16_coco/blk23 \
      --layer 23 --image_list /path/to/coco_train_5k.txt --skip_existing
"""

import os, sys, time, argparse, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms as tvtf
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", category=UserWarning)

DINOV3_DIR = Path("/data4/workspace/zlt/fasterrcnn-pytorch-training-pipeline/dinov3")
BACKBONE_WEIGHTS = DINOV3_DIR / "weights" / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_NORMALIZE = tvtf.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

DEFAULT_LAYER = 5
PATCH_SIZE = 16


def _resize_shortside(orig_w, orig_h, target=1024, max_size=1536):
    """Replicate training's get_size_with_aspect_ratio logic."""
    w, h = orig_w, orig_h
    size = target
    min_original_size = float(min(w, h))
    max_original_size = float(max(w, h))
    if max_original_size / min_original_size * size > max_size:
        size = int(round(max_size * min_original_size / max_original_size))
    if (w <= h and w == size) or (h <= w and h == size):
        return w, h
    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)
    return ow, oh


def build_backbone(device):
    sys.path.insert(0, str(DINOV3_DIR))
    backbone = torch.hub.load(
        str(DINOV3_DIR), "dinov3_vitl16",
        source="local", weights=str(BACKBONE_WEIGHTS),
    )
    return backbone.to(device).eval()


@torch.no_grad()
def extract_layer(backbone, img_path, device, layer):
    """Extract features at the given block layer with shorter-side=1024 resize.

    Returns:
        all_tokens: [T, D] numpy, CLS + reg + patches
        orig_size: (orig_h, orig_w)
        feat_H, feat_W: spatial dims of patch tokens
    """
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    new_w, new_h = _resize_shortside(orig_w, orig_h)
    img = img.resize((new_w, new_h), Image.BILINEAR)

    img_tensor = IMG_NORMALIZE(tvtf.ToTensor()(img)).unsqueeze(0).to(device)
    x, (H, W) = backbone.prepare_tokens_with_masks(img_tensor)
    for i in range(layer + 1):
        rope = backbone.rope_embed(H=H, W=W) if backbone.rope_embed else None
        x = backbone.blocks[i](x, rope)

    all_tokens = x.squeeze(0).cpu().numpy().astype(np.float32)
    return all_tokens, (orig_h, orig_w), H, W


def select_images(coco_root, n_images, seed, filter_w=None, filter_h=None):
    """Select n_images from COCO train2017, optionally filtered by resolution."""
    ann_file = os.path.join(coco_root, "annotations", "instances_train2017.json")
    with open(ann_file) as f:
        coco = json.load(f)
    all_images = coco["images"]
    if filter_w is not None and filter_h is not None:
        all_images = [img for img in all_images
                      if img["width"] == filter_w and img["height"] == filter_h]
        print(f"  After filtering {filter_w}x{filter_h}: {len(all_images)} images")
    rng = np.random.RandomState(seed)
    selected_idx = rng.choice(len(all_images), min(n_images, len(all_images)), replace=False)
    selected = [all_images[i] for i in sorted(selected_idx)]
    return selected


def main():
    p = argparse.ArgumentParser("DINOv3 COCO train2017 特征提取")
    p.add_argument("--coco_root", default="/data4/workspace/zlt/compression_vit/data/coco")
    p.add_argument("--out_root", default="/data4/workspace/zlt/featcodec/features/train/dinov3_vitl16_coco/blk05")
    p.add_argument("--layer", type=int, default=DEFAULT_LAYER,
                   help="Block index to extract (0-based, default=5 for blk05; 23 for last layer)")
    p.add_argument("--n_images", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--filter_w", type=int, default=None, help="Filter by image width")
    p.add_argument("--filter_h", type=int, default=None, help="Filter by image height")
    p.add_argument("--image_list", default=None, help="Pre-generated image list (12-digit IDs per line)")
    p.add_argument("--list_name", default="coco_train_5k.txt", help="Output image list filename")
    args = p.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.out_root, exist_ok=True)

    meta_dir = os.path.join(os.path.dirname(args.out_root), "meta")
    os.makedirs(meta_dir, exist_ok=True)

    if args.image_list and os.path.isfile(args.image_list):
        print(f"Loading pre-generated image list: {args.image_list}")
        ann_file = os.path.join(args.coco_root, "annotations", "instances_train2017.json")
        with open(ann_file) as f:
            coco = json.load(f)
        id_to_img = {img["id"]: img for img in coco["images"]}
        with open(args.image_list) as f:
            img_ids = [int(line.strip()) for line in f if line.strip()]
        images = [id_to_img[iid] for iid in img_ids if iid in id_to_img]
        print(f"  Loaded {len(images)} images from list")
    else:
        print(f"Selecting {args.n_images} images from COCO train2017 (seed={args.seed}) ...")
        images = select_images(args.coco_root, args.n_images, args.seed,
                               filter_w=args.filter_w, filter_h=args.filter_h)
        print(f"  Selected {len(images)} images")

    img_list_path = os.path.join(os.path.dirname(args.out_root), args.list_name)
    with open(img_list_path, "w") as f:
        for img_info in images:
            f.write(f"{img_info['id']:012d}\n")
    print(f"  Image list → {img_list_path}")

    print(f"Loading DINOv3 ViT-L/16 backbone ...")
    backbone = build_backbone(device)
    n_blocks = len(backbone.blocks)
    if not (0 <= args.layer < n_blocks):
        raise ValueError(f"--layer must be in [0, {n_blocks - 1}], got {args.layer}")
    layer_tag = f"blk{args.layer:02d}"
    print(f"  Extracting {layer_tag} (block {args.layer}/{n_blocks - 1})")

    t0 = time.time()
    n_done, n_skip = 0, 0
    img_dir = os.path.join(args.coco_root, "train2017")

    for img_info in tqdm(images, desc=f"Extracting {layer_tag}"):
        img_id = img_info["id"]
        img_id_str = f"{img_id:012d}"
        save_path = os.path.join(args.out_root, f"{img_id_str}.npy")
        meta_path = os.path.join(meta_dir, f"{img_id_str}.npz")

        if args.skip_existing and os.path.isfile(save_path):
            n_skip += 1
            continue

        img_path = os.path.join(img_dir, img_info["file_name"])
        if not os.path.isfile(img_path):
            print(f"[warn] missing: {img_path}")
            continue

        try:
            all_tokens, orig_size, feat_H, feat_W = extract_layer(
                backbone, img_path, device, args.layer)
        except Exception as e:
            print(f"[warn] failed {img_path}: {e}")
            continue

        np.save(save_path, all_tokens)

        np.savez_compressed(meta_path,
            orig_size=np.array(orig_size, dtype=np.int32),
            feat_hw=np.array([feat_H, feat_W], dtype=np.int32),
            n_tokens=np.array([all_tokens.shape[0]], dtype=np.int32),
            layer=np.array([args.layer], dtype=np.int32),
        )

        n_done += 1

    elapsed = time.time() - t0
    print(f"\nDone. saved={n_done} skipped={n_skip}")
    print(f"  out_root: {args.out_root}")
    print(f"  meta_dir: {meta_dir}")
    print(f"  Time: {elapsed:.1f}s ({n_done / max(elapsed, 1):.1f} img/s)")


if __name__ == "__main__":
    main()

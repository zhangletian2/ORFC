"""
Pre-crop segmentation features into (256, 256) patches for faster training I/O.

Input:  (2, 1370, 1024) per .npy  → ~10.7 MB
Output: (256, 256) per .npy       → ~0.25 MB

Usage (single layer):
    python tools/precrop_seg_features.py \
        --src features/voc2012_5000/dinov2_vitl14/blk20 \
        --dst features/voc2012_5000_crops/train/dinov2_vitl14/blk20 \
        --num_crops 50 --seed 42

Usage (batch layers):
    python tools/precrop_seg_features.py \
        --base_src features/voc2012_5000/dinov2_vitl14 \
        --base_dst features/voc2012_5000_crops/train/dinov2_vitl14 \
        --layers blk05 blk10 blk15 \
        --num_crops 50 --seed 42 --workers 8
"""
import argparse
import os
import glob
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def pack_seg_feature(feat):
    """(num_slides, 1+N, D) → (1+N, num_slides*D)"""
    return np.concatenate([feat[s] for s in range(feat.shape[0])], axis=-1)


def random_crop_2d(feat, crop_h, crop_w, rng):
    h, w = feat.shape
    y = rng.randint(0, h - crop_h + 1)
    x = rng.randint(0, w - crop_w + 1)
    return feat[y:y + crop_h, x:x + crop_w]


def process_one_file(fpath, dst_dir, num_crops, crop_h, crop_w, seed):
    rng = np.random.RandomState(seed)
    feat = np.load(fpath).astype(np.float32)
    packed = pack_seg_feature(feat)
    stem = Path(fpath).stem
    for j in range(num_crops):
        patch = random_crop_2d(packed, crop_h, crop_w, rng)
        np.save(os.path.join(dst_dir, f"{stem}_c{j:03d}.npy"), patch)
    return num_crops


def process_layer(src_dir, dst_dir, num_crops, crop_size, seed, workers):
    os.makedirs(dst_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(src_dir, "*.npy")))
    crop_h, crop_w = crop_size
    print(f"[{Path(src_dir).name}] {len(files)} files × {num_crops} crops = {len(files) * num_crops} patches, workers={workers}")

    # per-file seed derived from global seed + file index
    file_seeds = [seed + i for i in range(len(files))]
    total_saved = 0

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(process_one_file, f, dst_dir, num_crops, crop_h, crop_w, s): i
            for i, (f, s) in enumerate(zip(files, file_seeds))
        }
        for future in as_completed(futures):
            total_saved += future.result()
            if total_saved % (500 * num_crops) == 0:
                print(f"  [{Path(src_dir).name}] saved {total_saved} patches")

    print(f"[{Path(src_dir).name}] Done. Total: {total_saved} patches")
    return total_saved


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, default=None, help="Source dir (single layer mode)")
    parser.add_argument("--dst", type=str, default=None, help="Destination dir (single layer mode)")
    parser.add_argument("--base_src", type=str, default=None, help="Base source dir (batch mode)")
    parser.add_argument("--base_dst", type=str, default=None, help="Base destination dir (batch mode)")
    parser.add_argument("--layers", type=str, nargs="+", default=None, help="Layers to process (batch mode)")
    parser.add_argument("--num_crops", type=int, default=50)
    parser.add_argument("--crop_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.layers and args.base_src and args.base_dst:
        for layer in args.layers:
            src = os.path.join(args.base_src, layer)
            dst = os.path.join(args.base_dst, layer)
            process_layer(src, dst, args.num_crops, args.crop_size, args.seed, args.workers)
    elif args.src and args.dst:
        process_layer(args.src, args.dst, args.num_crops, args.crop_size, args.seed, args.workers)
    else:
        parser.error("Provide either --src/--dst (single) or --base_src/--base_dst/--layers (batch)")


if __name__ == "__main__":
    main()

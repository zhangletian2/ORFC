#!/usr/bin/env python3
"""
Verify FrozenTail manual cu_seqlens + RoPE reconstruction vs full visual.forward().

Compares two approaches:
  A) Full visual.forward(): load image → patch_embed → all 24 blocks → extract block[23] output
  B) FrozenTail: extract block[5] output → manually reconstruct cu_seqlens/RoPE → run blocks[6:23]

If FrozenTail correctly reimplements position info, A == B (numerically identical).

Usage:
    python verify_frozen_tail.py --num_images 5
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

MODEL_PATH = "/data4/workspace/zlt/cache/hf/hub/Qwen3-VL-4B-Thinking"
IMAGE_ROOT = "/data4/workspace/zlt/featcodec/ORFC/data/imagenet/images/val"
LIST_PATH = "/data4/workspace/zlt/featcodec/ORFC/utils/imagenet_selected_pathname500.txt"


def load_list(path):
    pairs = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                pairs.append((parts[0], parts[1]))
    return pairs


def prepare_pixel_values(processor, img):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": "."},
    ]}]
    inputs = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True,
        return_tensors="pt", add_generation_prompt=False,
    )
    return inputs["pixel_values"], inputs["image_grid_thw"]


def verify(args):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    device = args.device
    layer = args.layer
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    torch_dtype = dtype_map[args.dtype]

    print(f"Loading Qwen3-VL model (dtype={args.dtype}, device={device})...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch_dtype, device_map=device
    ).eval()
    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    visual = model.model.visual

    n_blocks = len(visual.blocks)
    print(f"  Visual tower: {n_blocks} blocks, spatial_merge_size={visual.spatial_merge_size}")
    print(f"  Extracting at layer={layer}, tail = blocks[{layer+1}:{n_blocks}]")

    # Load image list
    pairs = load_list(LIST_PATH)[:args.num_images]
    print(f"\nTesting {len(pairs)} images...\n")

    # ========================================================
    # Method A: Full visual.forward() with hooks
    # ========================================================
    # Hook block[layer] to get intermediate features
    # Hook block[-1] (block[23]) to get final output
    # ========================================================

    results_a = []  # block[23] output via full forward
    results_mid = []  # block[layer] output (intermediate features)
    results_grid = []  # grid_thw for each image

    class CaptureHook:
        def __init__(self, block):
            self.output = None
            self._handle = block.register_forward_hook(self._hook)

        def _hook(self, _mod, _inp, out):
            self.output = out.detach().clone()

        def close(self):
            self._handle.remove()

    hook_mid = CaptureHook(visual.blocks[layer])
    hook_last = CaptureHook(visual.blocks[n_blocks - 1])

    print("=" * 60)
    print("  Method A: Full visual.forward()")
    print("=" * 60)

    for i, (wnid, basename) in enumerate(pairs):
        img_path = os.path.join(IMAGE_ROOT, wnid, basename + ".JPEG")
        if not os.path.exists(img_path):
            print(f"  [{i}] SKIP (not found): {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")
        pv, grid = prepare_pixel_values(processor, img)
        pv = pv.to(device=device, dtype=torch_dtype)
        grid = grid.to(device=device)

        with torch.no_grad():
            _ = visual(pv, grid_thw=grid)

        out_mid = hook_mid.output  # [T, D] - block[layer] output
        out_last = hook_last.output  # [T, D] - block[23] output

        results_a.append(out_last.cpu().float())
        results_mid.append(out_mid.cpu().float())
        results_grid.append(grid.cpu())

        print(f"  [{i}] {basename}: T={out_mid.shape[0]}, grid={grid[0].tolist()}")

    hook_mid.close()
    hook_last.close()

    # ========================================================
    # Method B: FrozenTail manual reconstruction
    # ========================================================
    # Take block[layer] output from Method A, then manually
    # reconstruct cu_seqlens + RoPE and run blocks[layer+1:]
    # ========================================================

    print(f"\n{'=' * 60}")
    print(f"  Method B: FrozenTail (manual cu_seqlens + RoPE)")
    print(f"{'=' * 60}")

    tail_blocks = list(visual.blocks[layer + 1:])
    rotary_pos_emb_module = visual.rotary_pos_emb
    spatial_merge_size = visual.spatial_merge_size

    results_b = []

    for i, (mid_feat, grid) in enumerate(zip(results_mid, results_grid)):
        mid_feat = mid_feat.to(device=device, dtype=torch_dtype)
        grid = grid.to(device=device)

        # Reconstruct cu_seqlens + position_embeddings (same as FrozenTail._compute_position_info)
        cu_seqlens = torch.repeat_interleave(
            grid[:, 1] * grid[:, 2], grid[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        grid_thw_list = grid.tolist()
        max_hw = max(max(h, w) for _, h, w in grid_thw_list)
        freq_table = rotary_pos_emb_module(max_hw)

        total_tokens = sum(t * h * w for t, h, w in grid_thw_list)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw_list:
            merged_h = height // spatial_merge_size
            merged_w = width // spatial_merge_size
            block_rows = torch.arange(merged_h, device=device)
            block_cols = torch.arange(merged_w, device=device)
            intra_row = torch.arange(spatial_merge_size, device=device)
            intra_col = torch.arange(spatial_merge_size, device=device)

            row_idx = (block_rows[:, None, None, None] * spatial_merge_size
                       + intra_row[None, None, :, None])
            col_idx = (block_cols[None, :, None, None] * spatial_merge_size
                       + intra_col[None, None, None, :])
            row_idx = row_idx.expand(merged_h, merged_w, spatial_merge_size,
                                     spatial_merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, spatial_merge_size,
                                     spatial_merge_size).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset:offset + num_tokens] = coords
            offset += num_tokens

        rotary_pos_emb = freq_table[pos_ids].flatten(1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # Run tail blocks
        x = mid_feat
        with torch.no_grad():
            for blk in tail_blocks:
                x = blk(x, cu_seqlens=cu_seqlens,
                        position_embeddings=position_embeddings)

        results_b.append(x.cpu().float())

    # ========================================================
    # Compare results
    # ========================================================
    print(f"\n{'=' * 60}")
    print(f"  Comparison: Method A (full forward) vs Method B (FrozenTail)")
    print(f"{'=' * 60}")
    print(f"  {'Image':<40} {'Shape':<12} {'MaxAbsDiff':<14} {'RelDiff':<12} {'Match'}")
    print(f"  {'-'*40} {'-'*12} {'-'*14} {'-'*12} {'-'*6}")

    all_match = True
    for i, (wnid, basename) in enumerate(pairs[:len(results_a)]):
        a = results_a[i]
        b = results_b[i]

        max_abs_diff = (a - b).abs().max().item()
        norm_a = a.norm().item()
        rel_diff = max_abs_diff / (norm_a + 1e-8)

        # bf16 tolerance: ~1e-3 for accumulated operations
        match = max_abs_diff < 1e-2
        if not match:
            all_match = False

        status = "OK" if match else "FAIL"
        print(f"  {basename:<40} {str(list(a.shape)):<12} {max_abs_diff:<14.6e} {rel_diff:<12.6e} {status}")

    print(f"\n  {'='*60}")
    if all_match:
        print(f"  PASS: All {len(results_a)} images match (MaxAbsDiff < 1e-2)")
        print(f"  FrozenTail manual reconstruction is CORRECT.")
    else:
        print(f"  FAIL: Some images have significant differences!")
        print(f"  FrozenTail reconstruction may have bugs.")
    print(f"  {'='*60}")

    # Detailed statistics
    all_abs = [((results_a[i] - results_b[i]).abs().max().item()) for i in range(len(results_a))]
    print(f"\n  MaxAbsDiff stats: min={min(all_abs):.2e}, max={max(all_abs):.2e}, "
          f"mean={np.mean(all_abs):.2e}")

    # Check if differences are just floating point (bf16 accumulation)
    if all_match and max(all_abs) < 1e-5:
        print(f"  Differences are within bf16 numerical precision (< 1e-5).")
        print(f"  The two methods produce IDENTICAL results up to floating point.")
    elif all_match:
        print(f"  Differences likely from bf16 accumulation across {n_blocks - layer - 1} blocks.")
        print(f"  Functionally equivalent.")


def main():
    parser = argparse.ArgumentParser(description="Verify FrozenTail vs full visual.forward()")
    parser.add_argument("--num_images", type=int, default=5)
    parser.add_argument("--layer", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()
    verify(args)


if __name__ == "__main__":
    main()

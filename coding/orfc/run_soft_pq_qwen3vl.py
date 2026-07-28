#!/usr/bin/env python
"""
Differentiable Soft-PQ for Qwen3-VL-4B visual features.

Key differences from run_soft_pq.py:
  - Handles variable-length token sequences (Qwen3-VL dynamic resolution)
  - Supports both MSE and ΔL_ref training (with saved grid_thw metadata)
  - Downstream evaluation via MMBench replay pipeline
  - Rate evaluation on test features (per-token, length-agnostic)

Training: ImageNet features (train/qwen3vl_4b/blk05)
Evaluation: MMBench features → codec decode → qwen3vl_feat_pipeline.py replay

Usage:
    python run_soft_pq_qwen3vl.py --layer blk05 --K 64 --embedding_dim 32 --epochs 100

    # With ΔL_ref loss (requires grid_thw.json in feature dirs):
    python run_soft_pq_qwen3vl.py --layer blk05 --K 64 --epochs 100 --loss delta_l_ref \
        --model_path /path/to/Qwen3-VL-4B-Thinking

    # Quick smoke test:
    python run_soft_pq_qwen3vl.py --layer blk05 --K 64 --epochs 5 --max_train_images 50

    # Eval-only with existing codec:
    python run_soft_pq_qwen3vl.py --eval_only --ckpt_path checkpoints/qwen3vl_4b/blk05_K64_...pt
"""

import os, sys, argparse, json, math, time
import numpy as np
import torch
import torch.nn.functional as F
import subprocess
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

ORFC_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.normpath(os.path.join(ORFC_ROOT, "..", ".."))
if ORFC_ROOT not in sys.path:
    sys.path.insert(0, ORFC_ROOT)

from opq import (
    batch_normalize_gpu, batch_inv_normalize_gpu, batched_assign,
    learn_opq_rotation,
)
from soft_pq import (
    SoftPQ, FeatureTransform, OrthogonalTransform, FeatureCodec,
    save_codec, load_codec,
)

import warnings
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")

try:
    from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
    from compressai import ans as _ans
    _HAS_ANS = True
except (ImportError, ModuleNotFoundError):
    _HAS_ANS = False


# ================================================================
#                    Qwen3-VL FrozenTail (delta-L_ref support)
# ================================================================

class Qwen3VLFrozenTail:
    """Frozen tail blocks of Qwen3-VL visual tower for delta-L_ref computation.

    Replicates the forward pass of blocks[layer+1:] with proper
    cu_seqlens and position_embeddings reconstructed from grid_thw.

    The visual tower processes packed sequences: hidden_states is [seq_len, D]
    (not batched), and cu_seqlens marks image boundaries.
    """

    def __init__(self, model_path, layer, device='cuda', dtype='bf16',
                 grad_checkpoint=False, ckpt_segments=3):
        from transformers import Qwen3VLForConditionalGeneration
        torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                       "fp32": torch.float32}[dtype]
        self.grad_checkpoint = grad_checkpoint
        self.ckpt_segments = ckpt_segments

        print(f"  Loading Qwen3-VL visual tail (blocks[{layer+1}:])...")
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch_dtype, device_map='cpu'
        )
        visual = model.model.visual

        self.tail_blocks = list(visual.blocks[layer + 1:])
        self.rotary_pos_emb_module = visual.rotary_pos_emb
        self.spatial_merge_size = visual.spatial_merge_size
        self.device = device
        self.dtype = torch_dtype

        for blk in self.tail_blocks:
            blk.to(device).eval()
            for p in blk.parameters():
                p.requires_grad_(False)
        self.rotary_pos_emb_module.to(device)

        del model, visual
        torch.cuda.empty_cache()

        n_tail = len(self.tail_blocks)
        ckpt_info = ""
        if grad_checkpoint:
            seg = ckpt_segments if ckpt_segments > 0 else n_tail
            ckpt_info = f", ckpt={seg}seg ({n_tail // seg}blk/seg)"
        print(f"  Tail: {n_tail} blocks (blk{layer+1}..blk{layer+n_tail})"
              f"{ckpt_info}")

    def _compute_position_info(self, grid_thw):
        """Reconstruct cu_seqlens and position_embeddings from grid_thw.

        Args:
            grid_thw: Tensor [num_images, 3] or list of [t, h, w]

        Returns:
            cu_seqlens: [num_images + 1] int32 tensor
            position_embeddings: (cos, sin) each [seq_len, dim]
        """
        if isinstance(grid_thw, (list, tuple)):
            grid_thw = torch.tensor(grid_thw, dtype=torch.long, device=self.device)
        if grid_thw.dim() == 1:
            grid_thw = grid_thw.unsqueeze(0)

        grid_thw = grid_thw.to(self.device)
        merge_size = self.spatial_merge_size

        cu_seqlens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        ).cumsum(dim=0, dtype=torch.int32)
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        grid_thw_list = grid_thw.tolist()
        max_hw = max(max(h, w) for _, h, w in grid_thw_list)
        freq_table = self.rotary_pos_emb_module(max_hw)

        total_tokens = sum(t * h * w for t, h, w in grid_thw_list)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long,
                              device=self.device)

        offset = 0
        for num_frames, height, width in grid_thw_list:
            merged_h = height // merge_size
            merged_w = width // merge_size
            block_rows = torch.arange(merged_h, device=self.device)
            block_cols = torch.arange(merged_w, device=self.device)
            intra_row = torch.arange(merge_size, device=self.device)
            intra_col = torch.arange(merge_size, device=self.device)

            row_idx = (block_rows[:, None, None, None] * merge_size
                       + intra_row[None, None, :, None])
            col_idx = (block_cols[None, :, None, None] * merge_size
                       + intra_col[None, None, None, :])
            row_idx = row_idx.expand(merged_h, merged_w, merge_size,
                                     merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size,
                                     merge_size).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset:offset + num_tokens] = coords
            offset += num_tokens

        rotary_pos_emb = freq_table[pos_ids].flatten(1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        return cu_seqlens, position_embeddings

    def _run_segment(self, blocks, x, cu_seqlens, position_embeddings):
        """Run a list of blocks sequentially (used as checkpoint unit)."""
        for blk in blocks:
            x = blk(x, cu_seqlens=cu_seqlens,
                    position_embeddings=position_embeddings)
        return x

    def __call__(self, hidden_states, grid_thw):
        """Forward through tail blocks with segment-level gradient checkpointing.

        When grad_checkpoint=True, groups blocks into ckpt_segments segments.
        Each segment is checkpointed as a unit, so backward only recomputes
        blocks within each segment (not all 18).

        Args:
            hidden_states: [seq_len, D] tensor (packed, NOT batched)
            grid_thw: [num_images, 3] or list of [t, h, w]
        """
        cu_seqlens, position_embeddings = self._compute_position_info(grid_thw)
        x = hidden_states

        if self.grad_checkpoint and x.requires_grad:
            n = len(self.tail_blocks)
            seg = self.ckpt_segments if self.ckpt_segments > 0 else n
            seg_size = max(1, n // seg)
            for start in range(0, n, seg_size):
                segment = self.tail_blocks[start:start + seg_size]
                x = torch.utils.checkpoint.checkpoint(
                    self._run_segment, segment, x,
                    cu_seqlens, position_embeddings,
                    use_reentrant=False,
                )
        else:
            for blk in self.tail_blocks:
                x = blk(x, cu_seqlens=cu_seqlens,
                        position_embeddings=position_embeddings)
        return x

    @torch.no_grad()
    def forward_nograd(self, hidden_states, grid_thw):
        """No-grad forward for teacher computation."""
        return self(hidden_states, grid_thw)

    def to(self, device_or_str):
        """Move tail blocks to device (for memory management)."""
        if isinstance(device_or_str, str):
            device_or_str = torch.device(device_or_str)
        for blk in self.tail_blocks:
            blk.to(device_or_str)
        self.rotary_pos_emb_module.to(device_or_str)
        return self


# ================================================================
#                    Variable-length Feature Handling
# ================================================================

def preload_features_varlen(feat_dir, max_images=0, seed=42, load_grid_thw=False):
    """Load variable-length .npy features as a list (no stacking).

    Returns:
        features: list of [T_i, D] numpy arrays (float32)
        basenames: list of str
        grid_thws: list of [t, h, w] (only if load_grid_thw=True, else None)
    """
    feat_dir = Path(feat_dir)
    feat_files = sorted(feat_dir.glob("*.npy"))
    if not feat_files:
        raise FileNotFoundError(f"No .npy files in {feat_dir}")

    # Load grid_thw metadata if requested
    grid_thw_map = None
    if load_grid_thw:
        grid_thw_path = feat_dir / "grid_thw.json"
        if grid_thw_path.exists():
            with open(grid_thw_path) as f:
                grid_thw_map = json.load(f)
        else:
            print(f"  [WARN] grid_thw.json not found in {feat_dir}, "
                  f"delta-L_ref will fall back to MSE")

    if max_images > 0 and len(feat_files) > max_images:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(feat_files), max_images, replace=False)
        feat_files = [feat_files[i] for i in sorted(idx)]

    features = []
    basenames = []
    grid_thws = [] if grid_thw_map is not None else None
    for f in feat_files:
        features.append(np.load(f).astype(np.float32))
        bn = f.stem
        basenames.append(bn)
        if grid_thw_map is not None:
            thw = grid_thw_map.get(bn)
            if thw is None:
                thw = [1, 0, 0]  # placeholder, will be skipped
            grid_thws.append(thw)

    return features, basenames, grid_thws


class VarLenFeatureDataset(Dataset):
    """Dataset for variable-length features. Returns one image at a time."""

    def __init__(self, features, grid_thws=None, teacher_cache=None):
        self.features = features
        self.grid_thws = grid_thws
        self.teacher_cache = teacher_cache

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.features[idx]).float()
        if self.grid_thws is not None:
            thw = torch.tensor(self.grid_thws[idx], dtype=torch.long)
            if self.teacher_cache is not None:
                tc = torch.from_numpy(self.teacher_cache[idx]).float()
                return x, thw, tc
            return x, thw
        return x


def varlen_collate_mse(batch):
    """Collate for MSE mode: normalize per-image then concatenate.

    Returns:
        Y_cat: [1, sum_T, D] normalized and concatenated tensor
        lengths: [B] token count per image
        None, None (grid_thw / teacher placeholders)
    """
    eps = 1e-5
    norm_chunks = []
    lengths = []
    for x in batch:
        mu = x.mean()
        std = ((x - mu) ** 2).mean().add(eps).sqrt()
        norm_chunks.append((x - mu) / std)
        lengths.append(x.shape[0])
    Y_cat = torch.cat(norm_chunks, dim=0).unsqueeze(0)  # [1, sum_T, D]
    return Y_cat, torch.tensor(lengths, dtype=torch.long), None, None


def varlen_collate_dlref(batch):
    """Collate for delta-L_ref mode: return raw features + grid_thw + teacher.

    Each item is (feature_tensor, grid_thw_tensor[, teacher_tensor]).
    Returns:
        features_list: list of [T_i, D] tensors (unnormalized)
        lengths: [B] token count per image
        grid_thws: [B, 3] tensor of grid_thw
        teacher_list: list of [T_i, D'] tensors, or None
    """
    features_list = []
    lengths = []
    grid_thws = []
    teacher_list = []
    has_teacher = len(batch[0]) == 3
    for item in batch:
        if has_teacher:
            x, thw, tc = item
            teacher_list.append(tc)
        else:
            x, thw = item
        features_list.append(x)
        lengths.append(x.shape[0])
        grid_thws.append(thw)
    return (features_list,
            torch.tensor(lengths, dtype=torch.long),
            torch.stack(grid_thws),
            teacher_list if has_teacher else None)


# ================================================================
#                    Standard OPQ encode/decode (variable-length)
# ================================================================

def pq_encode_decode_varlen(features, codebooks, embedding_dim, norm_mode,
                            device, R=None):
    """OPQ encode/decode for variable-length features (one image at a time)."""
    num_groups = len(codebooks)
    cb_t = torch.from_numpy(np.stack(codebooks)).float().to(device)
    R_t = torch.from_numpy(R).float().to(device) if R is not None else None
    all_xhat = []
    for feat in features:
        C = feat.shape[1]
        X = torch.from_numpy(feat).float().unsqueeze(0).to(device)
        with torch.no_grad():
            Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
            flat = Y.reshape(-1, C)
            Z = flat @ R_t if R_t is not None else flat
            z_3d = Z.reshape(-1, num_groups, embedding_dim) \
                    .permute(1, 0, 2).contiguous()
            z_hat_3d, _ = batched_assign(z_3d, cb_t, device=device)
            flat_hat = z_hat_3d.permute(1, 0, 2).reshape(-1, C)
            Y_hat = flat_hat @ R_t.T if R_t is not None else flat_hat
            Y_hat = Y_hat.reshape(1, -1, C)
            X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
        all_xhat.append(X_hat[0].cpu().numpy())
        del X, Y, Mu, Std, Z, z_3d, z_hat_3d, flat_hat, Y_hat, X_hat
    torch.cuda.empty_cache()
    return all_xhat


def codec_encode_decode_varlen(features, codec, norm_mode, device):
    """Codec encode/decode for variable-length features."""
    codec.eval()
    all_xhat = []
    with torch.no_grad():
        for feat in features:
            X = torch.from_numpy(feat).float().unsqueeze(0).to(device)
            Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
            Y_hat, _ = codec(Y)
            X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)
            all_xhat.append(X_hat[0].cpu().numpy())
            del X, Y, Mu, Std, Y_hat, X_hat
    torch.cuda.empty_cache()
    return all_xhat


# ================================================================
#                    Rate evaluation
# ================================================================

def _codec_labels_varlen(features, codec, norm_mode, device):
    """Run codec on variable-length features and return hard labels [G, N_total_tokens]."""
    codec.eval()
    pq = codec.pq
    all_labels = []
    with torch.no_grad():
        for feat in features:
            X = torch.from_numpy(feat).float().unsqueeze(0).to(device)
            Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
            _ = codec(Y)
            all_labels.append(pq._last_labels.cpu())
            del X, Y
    return torch.cat(all_labels, dim=1).numpy()


def _histogram_pmf(labels, G, K, smoothing=1.0):
    """Build list of G PMFs from [G, N] label array."""
    pmfs = []
    for g in range(G):
        counts = np.zeros(K, dtype=np.float64)
        np.add.at(counts, labels[g], 1)
        counts += smoothing
        pmfs.append(counts / counts.sum())
    return pmfs


def _rans_encode_bpt(labels_np, pmf_list, G, K, precision=16):
    """Encode labels with rANS, return actual bits per token."""
    if not _HAS_ANS:
        return None
    encoder = _ans.RansEncoder()
    N = labels_np.shape[1]
    cdfs = []
    cdf_sizes = []
    for g in range(G):
        p = torch.from_numpy(pmf_list[g]).float()
        overflow = (1.0 - p.sum()).clamp_min(0)
        p = torch.cat([p, overflow.unsqueeze(0)])
        cdf = _pmf_to_quantized_cdf(p.tolist(), precision)
        cdfs.append(cdf)
        cdf_sizes.append(K + 2)
    symbols = []
    cdf_indices = []
    for n in range(N):
        for g in range(G):
            symbols.append(int(labels_np[g, n]))
            cdf_indices.append(g)
    byte_string = encoder.encode_with_indexes(
        symbols, cdf_indices, cdfs,
        cdf_sizes, [0] * G,
    )
    total_bits = len(byte_string) * 8
    return total_bits / N


def evaluate_rate(features, codec, norm_mode, device, train_pmf=None):
    """Compute rate metrics on variable-length features."""
    pq = codec.pq
    G = pq.G
    K = pq.K

    test_labels = _codec_labels_varlen(features, codec, norm_mode, device)
    N = test_labels.shape[1]

    if pq.use_rate:
        primary_pmf = pq.get_prior_pmf()
    elif train_pmf is not None:
        primary_pmf = train_pmf
    else:
        primary_pmf = np.full((G, K), 1.0 / K)

    xent_primary = 0.0
    for g in range(G):
        log2_p = np.log2(primary_pmf[g] + 1e-30)
        xent_primary += -log2_p[test_labels[g]].sum()
    xent_primary_bpt = xent_primary / N

    xent_train_bpt = None
    if train_pmf is not None:
        xent_train = 0.0
        for g in range(G):
            log2_t = np.log2(train_pmf[g] + 1e-30)
            xent_train += -log2_t[test_labels[g]].sum()
        xent_train_bpt = xent_train / N

    test_pmf = _histogram_pmf(test_labels, G, K, smoothing=0)
    empirical_entropy = 0.0
    for g in range(G):
        pg = test_pmf[g]
        pg = pg[pg > 0]
        empirical_entropy += -np.sum(pg * np.log2(pg))

    rans_bpt = _rans_encode_bpt(test_labels, primary_pmf, G, K)
    rans_train_bpt = None
    if train_pmf is not None:
        rans_train_bpt = _rans_encode_bpt(test_labels, train_pmf, G, K)

    max_rate = G * math.log2(K)
    result = {
        'xent_rate_bpt': float(xent_primary_bpt),
        'empirical_entropy_bpt': float(empirical_entropy),
        'max_rate_bpt': float(max_rate),
    }
    if xent_train_bpt is not None:
        result['xent_train_bpt'] = float(xent_train_bpt)
    if rans_bpt is not None:
        result['rans_bpt'] = float(rans_bpt)
    if rans_train_bpt is not None:
        result['rans_train_bpt'] = float(rans_train_bpt)
    return result


# ================================================================
#                    MSE evaluation
# ================================================================

def evaluate_mse_varlen(features, codec, norm_mode, device):
    """Compute per-token MSE in normalized space for variable-length features."""
    codec.eval()
    total_mse = 0.0
    total_tokens = 0
    with torch.no_grad():
        for feat in features:
            X = torch.from_numpy(feat).float().unsqueeze(0).to(device)
            Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
            Y_hat, _ = codec(Y)
            mse = ((Y - Y_hat) ** 2).sum().item()
            total_mse += mse
            total_tokens += feat.shape[0]
            del X, Y, Mu, Std, Y_hat
    return total_mse / total_tokens


def evaluate_dlref_varlen(features, grid_thws, teacher_cache,
                          tail, codec, norm_mode, device):
    """Compute per-token ΔL_ref using pre-computed teacher outputs."""
    codec.eval()
    total_dlref = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, feat in enumerate(features):
            X = torch.from_numpy(feat).float().unsqueeze(0).to(device)
            Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
            Y_hat, _ = codec(Y)
            X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)

            thw_t = torch.tensor(grid_thws[i], dtype=torch.long,
                                 device=device).unsqueeze(0)
            student = tail.forward_nograd(
                X_hat.squeeze(0).to(tail.dtype), thw_t)
            teacher = torch.from_numpy(teacher_cache[i]).float().to(device)

            total_dlref += ((teacher - student.float()) ** 2).sum().item()
            total_tokens += feat.shape[0]
            del X, Y, Mu, Std, Y_hat, X_hat, student, teacher
    return total_dlref / total_tokens


def evaluate_opq_dlref_varlen(features, grid_thws, teacher_cache,
                              tail, codebooks, R, embedding_dim, norm_mode, device):
    """Compute per-token ΔL_ref for OPQ baseline using pre-computed teachers."""
    G = len(codebooks)
    cb_t = torch.from_numpy(np.stack(codebooks)).float().to(device)
    R_t = torch.from_numpy(R).float().to(device) if R is not None else None
    total_dlref = 0.0
    total_tokens = 0
    with torch.no_grad():
        for i, feat in enumerate(features):
            C = feat.shape[1]
            X = torch.from_numpy(feat).float().unsqueeze(0).to(device)
            Y, Mu, Std = batch_normalize_gpu(X, mode=norm_mode)
            flat = Y.reshape(-1, C)
            Z = flat @ R_t if R_t is not None else flat
            z_3d = Z.reshape(-1, G, embedding_dim).permute(1, 0, 2).contiguous()
            z_hat_3d, _ = batched_assign(z_3d, cb_t, device=device)
            flat_hat = z_hat_3d.permute(1, 0, 2).reshape(-1, C)
            Y_hat = (flat_hat @ R_t.T if R_t is not None else flat_hat).reshape(1, -1, C)
            X_hat = batch_inv_normalize_gpu(Y_hat, Mu, Std)

            thw_t = torch.tensor(grid_thws[i], dtype=torch.long,
                                 device=device).unsqueeze(0)
            student = tail.forward_nograd(
                X_hat.squeeze(0).to(tail.dtype), thw_t)
            teacher = torch.from_numpy(teacher_cache[i]).float().to(device)

            total_dlref += ((teacher - student.float()) ** 2).sum().item()
            total_tokens += feat.shape[0]
            del X, Y, Mu, Std, flat, Z, z_3d, z_hat_3d, flat_hat, Y_hat, X_hat, student, teacher
    del cb_t, R_t
    torch.cuda.empty_cache()
    return total_dlref / total_tokens


# ================================================================
#                    MMBench evaluation via replay
# ================================================================

def evaluate_mmbench(decoded_dir, args):
    """Run MMBench replay evaluation via qwen3vl_feat_pipeline.py.

    Returns accuracy (float) or None if evaluation fails.
    """
    replay_script = os.path.join(PROJECT_ROOT, "tools", "qwen3vl_feat_pipeline.py")
    if not os.path.exists(replay_script):
        print(f"  [WARN] replay script not found: {replay_script}")
        return None

    data_dir = getattr(args, 'mmbench_data_dir',
                       os.path.join(PROJECT_ROOT, "data", "MMBench"))
    python_exec = args.python_exec
    out_json = os.path.join(decoded_dir, "replay_results.json")

    cmd = [
        python_exec, replay_script,
        "--model_path", args.model_path,
        "--data_dir", data_dir,
        "--max_new_tokens", "2048",
    ]
    if args.mmbench_sample_list:
        cmd += ["--sample_list", args.mmbench_sample_list]
    cmd += [
        "replay",
        "--layer", str(args.vit_layer),
        "--feat_dir", decoded_dir,
        "--out", out_json,
    ]

    print(f"  Running MMBench replay: {' '.join(cmd[-6:])}")
    try:
        replay_timeout = getattr(args, 'replay_timeout', 86400)
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=replay_timeout,
        )
        if result.returncode != 0:
            print(f"  [WARN] replay failed (rc={result.returncode})")
            if result.stderr:
                for line in result.stderr.strip().split('\n')[-5:]:
                    print(f"    {line}")
            return None
    except subprocess.TimeoutExpired:
        print(f"  [WARN] replay timed out")
        return None

    if not os.path.exists(out_json):
        print(f"  [WARN] replay output not found: {out_json}")
        return None

    with open(out_json) as f:
        data = json.load(f)
    acc = data.get("accuracy", data.get("acc"))
    if acc is not None:
        return float(acc)
    return None


# ================================================================
#                    Training loop (variable-length MSE)
# ================================================================

def train_soft_pq_varlen(
    features_train,
    G, K, d,
    norm_mode='per_image',
    epochs=100,
    lr=1e-3,
    batch_size=8,
    device='cuda',
    seed=42,
    val_features=None,
    verbose=True,
    transform=None,
    R_init=None,
    codebooks_init=None,
    kmeans_max_samples=2_000_000,
    lmbda=0.0,
    prior_init_counts=None,
    grad_clip=1.0,
    freeze_transform=False,
    freeze_codebooks=False,
    prior_floor=0.0,
    tau_start=1.0,
    tau_end=0.01,
    tau_schedule='exponential',
    warm_start_opq=False,
    tail=None,
    grid_thws_train=None,
    val_grid_thws=None,
):
    """Train Soft-PQ codec on variable-length features.

    Supports two loss modes:
      - MSE (tail=None): minimize ||Y - Y_hat||^2 in normalized space
      - delta-L_ref (tail provided): minimize ||tail(X) - tail(X_hat)||^2

    Loss is normalized per-token. ΔL_ref uses per-image gradient
    accumulation to limit peak GPU memory.

    For delta-L_ref, grid_thws_train must be provided (list of [t, h, w]).
    Teacher outputs are pre-computed once and cached in CPU memory.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    N_img = len(features_train)
    D = features_train[0].shape[1]

    pq = SoftPQ(G, K, d, lmbda=lmbda, prior_floor=prior_floor).to(device)
    if transform is not None:
        transform = transform.to(device)
    codec = FeatureCodec(pq, transform).to(device)

    _use_soft = (tau_start > 0)

    if R_init is not None and codebooks_init is not None and transform is not None:
        if verbose:
            print(f"  Warm-start from OPQ (transform + codebooks)")
        transform.init_from_opq(R_init)
        pq.init_codebooks(codebooks_init)
        if pq.use_rate and prior_init_counts is not None:
            pq.init_prior_from_freq(prior_init_counts)
            if verbose:
                print(f"  log_prior init from OPQ empirical frequency")
    elif codebooks_init is not None:
        if verbose:
            print(f"  Warm-start codebooks only")
        pq.init_codebooks(codebooks_init)
        if pq.use_rate and prior_init_counts is not None:
            pq.init_prior_from_freq(prior_init_counts)
    else:
        if verbose:
            print(f"  K-means init for codebooks ({N_img} images)...")
        all_Z = []
        for feat in features_train:
            X = torch.from_numpy(feat).float().unsqueeze(0).to(device)
            with torch.no_grad():
                Y, _, _ = batch_normalize_gpu(X, mode=norm_mode)
                flat = Y.reshape(-1, D)
                Z = transform.encode(flat) if transform else flat
            all_Z.append(Z.cpu())
            del X, Y, flat, Z
        Z_flat = torch.cat(all_Z, dim=0)
        del all_Z
        max_km = kmeans_max_samples
        if Z_flat.shape[0] > max_km:
            idx = np.random.choice(Z_flat.shape[0], max_km, replace=False)
            Z_flat = Z_flat[idx]
        pq.init_from_kmeans(Z_flat, device=device)
        del Z_flat
        torch.cuda.empty_cache()
        if verbose:
            print(f"  K-means init done.")

    if freeze_transform and transform is not None:
        for p in transform.parameters():
            p.requires_grad_(False)
        if verbose:
            n_p = sum(p.numel() for p in transform.parameters())
            print(f"  Frozen: transform ({n_p:,} params)")
    if freeze_codebooks:
        pq.codebooks.requires_grad_(False)
        if pq.use_rate:
            pq.log_prior.requires_grad_(False)
        if verbose:
            print(f"  Frozen: codebooks ({pq.codebooks.numel():,} params)")

    trainable = [p for p in codec.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )

    if verbose:
        n_trainable = sum(p.numel() for p in trainable)
        print(f"  Trainable params: {n_trainable:,}")
        if transform is not None and hasattr(transform, 'orth_error'):
            print(f"  Orthogonal transform: ||R'R-I||={transform.orth_error():.2e}")
        if _use_soft:
            print(f"  Soft PQ: tau {tau_start:.2f} -> {tau_end:.4f} ({tau_schedule})")
        else:
            print(f"  Hard PQ (tau=0)")

    use_dlref = (tail is not None and grid_thws_train is not None)

    # Pre-compute teacher outputs (ΔL_ref mode) - cached in CPU
    teacher_cache = None
    val_teacher_cache = None
    if use_dlref:
        if verbose:
            print(f"  Pre-computing teacher outputs ({N_img} images)...")
        t_pre = time.time()
        teacher_cache = []
        for feat in features_train:
            feat_t = torch.from_numpy(feat).to(device=device, dtype=tail.dtype)
            thw_t = torch.tensor(
                grid_thws_train[len(teacher_cache)], dtype=torch.long,
                device=device).unsqueeze(0)
            with torch.no_grad():
                out = tail.forward_nograd(feat_t, thw_t)
            teacher_cache.append(out.cpu().float().numpy())
            del feat_t, out
        torch.cuda.empty_cache()
        cache_mb = sum(tc.nbytes for tc in teacher_cache) / 1e6
        if verbose:
            print(f"  Teacher cache: {cache_mb:.0f} MB CPU ({time.time()-t_pre:.1f}s)")

        if val_features is not None and val_grid_thws is not None:
            if verbose:
                print(f"  Pre-computing val teacher outputs ({len(val_features)} images)...")
            val_teacher_cache = []
            for feat in val_features:
                feat_t = torch.from_numpy(feat).to(
                    device=device, dtype=tail.dtype)
                thw_t = torch.tensor(
                    val_grid_thws[len(val_teacher_cache)], dtype=torch.long,
                    device=device).unsqueeze(0)
                with torch.no_grad():
                    out = tail.forward_nograd(feat_t, thw_t)
                val_teacher_cache.append(out.cpu().float().numpy())
                del feat_t, out
            torch.cuda.empty_cache()

    if use_dlref:
        train_dataset = VarLenFeatureDataset(
            features_train, grid_thws=grid_thws_train,
            teacher_cache=teacher_cache)
        collate_fn = varlen_collate_dlref
        if verbose:
            print(f"  Loss: delta-L_ref (tail frozen, teacher cached)")
    else:
        train_dataset = VarLenFeatureDataset(features_train)
        collate_fn = varlen_collate_mse
        if verbose:
            print(f"  Loss: MSE (normalized space, per-token)")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0 if use_dlref else 2,
        pin_memory=True,
        persistent_workers=False,
        collate_fn=collate_fn,
    )
    history = []
    log_interval = max(1, epochs // 10)

    for epoch in range(epochs):
        t_epoch = time.time()

        if _use_soft and epochs > 1:
            progress = epoch / (epochs - 1)
            if tau_schedule == 'linear':
                tau = tau_start + (tau_end - tau_start) * progress
            else:
                tau = tau_start * (tau_end / tau_start) ** progress
            pq.temperature = tau
        elif _use_soft:
            pq.temperature = tau_start
        else:
            pq.temperature = 0.0

        total_distortion = 0.0
        total_rate_bits = 0.0
        total_tokens = 0
        usage_acc = torch.zeros(G, K, device=device)

        codec.train()
        for batch in train_loader:
            if use_dlref:
                features_list, lengths, grid_thws_batch, teacher_batch = batch
                B = len(lengths)
                batch_tokens = lengths.sum().item()

                optimizer.zero_grad()

                # Phase 1: per-image codec forward (small, fast)
                X_hats = []
                batch_usage = None
                rate_terms = []
                for i in range(B):
                    feat_i = features_list[i].to(device, non_blocking=True)
                    eps = 1e-5
                    mu = feat_i.mean()
                    std = ((feat_i - mu) ** 2).mean().add(eps).sqrt()
                    Y_i = (feat_i - mu) / std

                    Y_hat_i, usage_i = codec(Y_i.unsqueeze(0))
                    X_hat_i = Y_hat_i.squeeze(0) * std + mu
                    X_hats.append(X_hat_i)

                    if codec.use_rate:
                        rate_terms.append(codec._last_rate * feat_i.shape[0])
                    if batch_usage is None:
                        batch_usage = usage_i.detach()
                    else:
                        batch_usage += usage_i.detach()
                    del feat_i, Y_i, Y_hat_i

                # Phase 2: batched tail forward (pack B images, single call)
                X_hat_packed = torch.cat(X_hats, dim=0).to(tail.dtype)
                student_packed = tail(X_hat_packed, grid_thws_batch)

                # Phase 3: single loss + backward
                teachers_packed = torch.cat(
                    [t.to(device, non_blocking=True) for t in teacher_batch],
                    dim=0)
                total_dist = (
                    (teachers_packed - student_packed.float()) ** 2).sum()

                if codec.use_rate and rate_terms:
                    total_rate = sum(rate_terms)
                    loss = (lmbda * total_dist + total_rate) / batch_tokens
                else:
                    loss = total_dist / batch_tokens
                loss.backward()

                total_distortion += total_dist.item()
                if codec.use_rate:
                    total_rate_bits += sum(r.item() for r in rate_terms)
                total_tokens += batch_tokens

                del (X_hats, X_hat_packed, student_packed,
                     teachers_packed, total_dist, loss, rate_terms)

                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        codec.parameters(), grad_clip)
                optimizer.step()
                usage_acc += batch_usage

            else:
                # MSE mode: per-token normalized loss
                Y_cat, lengths_b, _, _ = batch
                Y_cat = Y_cat.to(device, non_blocking=True)
                sum_T = Y_cat.shape[1]

                Y_hat, usage = codec(Y_cat)
                sq_err = ((Y_cat - Y_hat) ** 2).sum()
                distortion = sq_err / sum_T

                if codec.use_rate:
                    loss = codec._last_rate + lmbda * distortion
                else:
                    loss = distortion

                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        codec.parameters(), grad_clip)
                optimizer.step()

                total_distortion += sq_err.item()
                if codec.use_rate:
                    total_rate_bits += codec._last_rate.item() * sum_T
                total_tokens += sum_T
                usage_acc += usage.detach()

                del Y_cat, Y_hat, loss, distortion, sq_err

        scheduler.step()

        avg_dist_pt = total_distortion / max(total_tokens, 1)
        avg_rate_bpt = (total_rate_bits / max(total_tokens, 1)
                        if codec.use_rate else 0.0)

        usage_norm = usage_acc / usage_acc.sum(
            dim=-1, keepdim=True).clamp(min=1e-12)
        log_p = torch.log(usage_norm + 1e-30)
        entropy = -(usage_norm * log_p).sum(dim=-1).mean().item()
        perplexity = math.exp(entropy)
        dead_entries = int((usage_acc == 0).sum().item())

        # Validation (matches training loss type)
        val_metric = None
        if val_features is not None and (epoch == 0 or epoch % log_interval == 0
                                         or epoch == epochs - 1):
            if use_dlref and val_teacher_cache is not None:
                val_metric = evaluate_dlref_varlen(
                    val_features, val_grid_thws, val_teacher_cache,
                    tail, codec, norm_mode, device)
            else:
                val_metric = evaluate_mse_varlen(
                    val_features, codec, norm_mode, device)

        epoch_time = time.time() - t_epoch
        val_label = ('val_dlref' if (use_dlref and val_teacher_cache)
                     else 'val_mse')
        record = {
            'epoch': epoch,
            'dist_pt': avg_dist_pt,
            'rate_bpt': avg_rate_bpt,
            'lr': optimizer.param_groups[0]['lr'],
            'temperature': pq.temperature,
            'perplexity': perplexity,
            'dead': dead_entries,
            'time': epoch_time,
        }
        if val_metric is not None:
            record[val_label] = val_metric
        if transform is not None and hasattr(transform, 'orth_error'):
            record['orth_error'] = transform.orth_error()
        history.append(record)

        if verbose and (epoch == 0 or epoch % log_interval == 0
                       or epoch == epochs - 1):
            tau_s = f"tau={pq.temperature:.4f}" if _use_soft else ""
            val_s = (f"  {val_label}={val_metric:.6f}"
                     if val_metric is not None else "")
            rate_s = (f"  rate={avg_rate_bpt:.2f}bpt"
                      if codec.use_rate else "")
            print(f"  [{epoch:3d}/{epochs}] dist/t={avg_dist_pt:.4f}"
                  f"{rate_s}  ppl={perplexity:.1f}  dead={dead_entries}"
                  f"  {tau_s}{val_s}  ({epoch_time:.1f}s)")

    return codec, history


# ================================================================
#                    Main experiment
# ================================================================

def compute_perplexity_from_usage(usage_acc):
    usage_norm = usage_acc / usage_acc.sum(dim=-1, keepdim=True).clamp(min=1e-12)
    log_p = torch.log(usage_norm + 1e-30)
    entropy = -(usage_norm * log_p).sum(dim=-1).mean().item()
    return math.exp(entropy)


def run_experiment(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    layer_idx = int(args.layer[-2:])

    print(f"\n{'#' * 70}")
    print(f"# Qwen3-VL Soft-PQ Feature Codec")
    print(f"# layer={args.layer} (idx={layer_idx}), K={args.K}, "
          f"emb={args.embedding_dim}, bt={args.bottleneck_dim}")
    print(f"# epochs={args.epochs}, lr={args.lr}, lmbda={args.lmbda}")
    tau_info = f", tau={args.tau_start}->{args.tau_end}" if args.tau_start > 0 else ""
    print(f"# warm_start={args.warm_start_opq}{tau_info}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}")

    # ---- Load features ----
    train_dir = Path(args.feat_root) / args.train_subset / args.backbone / args.layer

    use_dlref = (args.loss == 'delta_l_ref')
    print(f"\nLoading features... (loss={args.loss})")
    print(f"  train: {train_dir}")

    features_train, _, grid_thws_train = preload_features_varlen(
        train_dir, max_images=args.max_train_images, seed=args.seed,
        load_grid_thw=use_dlref)

    D = features_train[0].shape[1]
    token_counts_train = [f.shape[0] for f in features_train]
    bt_dim = args.bottleneck_dim
    Dp = bt_dim if bt_dim > 0 else D
    num_groups = Dp // args.embedding_dim
    bits_per_token = num_groups * math.log2(args.K)

    print(f"  train: {len(features_train)} images, "
          f"T={min(token_counts_train)}~{max(token_counts_train)} "
          f"(mean={np.mean(token_counts_train):.0f})")
    print(f"  D={D}, D'={Dp}, G={num_groups}, d={args.embedding_dim}")
    print(f"  bits/token={bits_per_token:.0f}")

    # Validation set
    n_val = min(args.n_val, len(features_train))
    rng_val = np.random.RandomState(args.seed + 1)
    val_idx = rng_val.choice(len(features_train), n_val, replace=False)
    val_features = [features_train[i] for i in val_idx]
    val_grid_thws = None
    if grid_thws_train is not None:
        val_grid_thws = [grid_thws_train[i] for i in val_idx]

    results = {}
    results['config'] = vars(args)

    # ================================================================
    #   (A) Standard OPQ baseline
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"  [Standard OPQ] baseline")
    print(f"{'=' * 60}")

    t0 = time.time()
    opq_groups = D // args.embedding_dim
    max_flat = args.kmeans_max_samples // opq_groups

    total_tokens = sum(f.shape[0] for f in features_train)
    sample_ratio = min(1.0, max_flat / total_tokens)
    rng_opq = np.random.RandomState(args.seed)

    sampled_vectors = []
    for feat in features_train:
        X = torch.from_numpy(feat).float().unsqueeze(0).to(device)
        with torch.no_grad():
            Y, _, _ = batch_normalize_gpu(X, mode=args.norm_mode)
            flat = Y.reshape(-1, D).cpu().numpy()
        n = flat.shape[0]
        n_keep = max(1, int(n * sample_ratio))
        if n_keep < n:
            idx_s = rng_opq.choice(n, n_keep, replace=False)
            sampled_vectors.append(flat[idx_s])
        else:
            sampled_vectors.append(flat)
        del X, Y, flat
    full_vectors = np.concatenate(sampled_vectors, axis=0)
    del sampled_vectors
    torch.cuda.empty_cache()

    print(f"  OPQ training: {full_vectors.shape[0]} vectors, "
          f"G={opq_groups}, K={args.K}, d={args.embedding_dim}")
    R_std, codebooks_std, hist_std = learn_opq_rotation(
        full_vectors, opq_groups, args.embedding_dim, args.K,
        max_iter_opq=20, max_iter_kmeans=100,
        device=device, verbose=False,
    )
    del full_vectors
    torch.cuda.empty_cache()
    std_time = time.time() - t0
    print(f"  OPQ done: MSE={hist_std[-1][0]:.8f} ({std_time:.1f}s)")

    # OPQ usage counts for prior init
    opq_usage_counts = None
    if args.lmbda > 0 and args.warm_start_opq:
        R_t = torch.from_numpy(R_std).float().to(device)
        cb_t = torch.from_numpy(np.stack(codebooks_std)).float().to(device)
        opq_usage_counts = np.zeros((opq_groups, args.K), dtype=np.float64)
        for feat in features_train:
            X = torch.from_numpy(feat).float().unsqueeze(0).to(device)
            with torch.no_grad():
                Y, _, _ = batch_normalize_gpu(X, mode=args.norm_mode)
                flat = Y.reshape(-1, D) @ R_t
                sub = flat.reshape(-1, opq_groups, args.embedding_dim) \
                      .permute(1, 0, 2).contiguous()
                dists = torch.cdist(sub, cb_t)
                labels = dists.argmin(dim=-1)
                for g in range(opq_groups):
                    for k in labels[g].cpu().numpy():
                        opq_usage_counts[g, k] += 1
            del X, Y, flat, sub, dists, labels
        del R_t, cb_t
        torch.cuda.empty_cache()
        ppl_opq = np.exp(-(opq_usage_counts / opq_usage_counts.sum(-1, keepdims=True)
                          * np.log(opq_usage_counts / opq_usage_counts.sum(-1, keepdims=True)
                                   + 1e-30)).sum(-1)).mean()
        print(f"  OPQ empirical ppl={ppl_opq:.1f}")

    # ================================================================
    #   (B) Codec training
    # ================================================================
    opq_val_dlref = None
    codec_val_dlref = None

    if getattr(args, 'eval_only', False) and args.ckpt_path:
        print(f"\n{'=' * 60}")
        print(f"  [Eval-Only] Loading codec from: {args.ckpt_path}")
        print(f"{'=' * 60}")
        codec = load_codec(args.ckpt_path, device=device)
        train_time = 0.0
        history = []
    else:
        bt_str = f"bt={bt_dim}" if bt_dim > 0 else "no transform"
        print(f"\n{'=' * 60}")
        tau_str = f", tau={args.tau_start}->{args.tau_end}" if args.tau_start > 0 else ""
        print(f"  [Codec] K={args.K}, lmbda={args.lmbda}, {bt_str}, "
              f"epochs={args.epochs}{tau_str}")
        print(f"{'=' * 60}")

        transform = None
        if bt_dim > 0 and bt_dim == D:
            transform = OrthogonalTransform(D)
        elif bt_dim > 0:
            transform = FeatureTransform(D, bt_dim)

        can_warmstart = (args.warm_start_opq and bt_dim == D)
        R_ws = None
        C_ws = None
        if can_warmstart:
            R_ws = R_std.copy()
            C_ws = [c.copy() for c in codebooks_std]
            if np.linalg.det(R_ws) < 0:
                R_ws[:, -1] *= -1
                C_ws[-1][:, -1] *= -1
                print(f"  det(R_opq)<0: flipped last col to SO(D)")

        if bt_dim > 0 and bt_dim != D:
            print(f"  Bottleneck D'={bt_dim} < D={D}: "
                  f"k-means init (OPQ warm-start N/A)")
            opq_usage_counts = None

        # Load tail for delta-L_ref if requested
        frozen_tail = None
        if use_dlref and grid_thws_train is not None:
            frozen_tail = Qwen3VLFrozenTail(
                args.model_path, layer_idx, device=str(device),
                dtype=args.tail_dtype,
                grad_checkpoint=args.grad_checkpoint,
                ckpt_segments=args.ckpt_segments)

        t0 = time.time()
        codec, history = train_soft_pq_varlen(
            features_train=features_train,
            G=num_groups,
            K=args.K,
            d=args.embedding_dim,
            norm_mode=args.norm_mode,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            seed=args.seed,
            val_features=val_features,
            verbose=True,
            transform=transform,
            R_init=R_ws,
            codebooks_init=C_ws,
            kmeans_max_samples=args.kmeans_max_samples,
            lmbda=args.lmbda,
            prior_init_counts=opq_usage_counts,
            grad_clip=args.grad_clip,
            freeze_transform=args.freeze_transform,
            freeze_codebooks=args.freeze_codebooks,
            prior_floor=args.prior_floor,
            tau_start=args.tau_start,
            tau_end=args.tau_end,
            tau_schedule=args.tau_schedule,
            warm_start_opq=args.warm_start_opq,
            tail=frozen_tail,
            grid_thws_train=grid_thws_train,
            val_grid_thws=val_grid_thws,
        )

        train_time = time.time() - t0
        print(f"  Codec training: {train_time:.1f}s")

        # Compute OPQ and Codec ΔL_ref on validation set (before releasing tail)
        opq_val_dlref = None
        codec_val_dlref = None
        if frozen_tail is not None and val_grid_thws is not None:
            print(f"\n  Computing ΔL_ref on validation set ({len(val_features)} images)...")
            val_teacher_cache = []
            for feat in val_features:
                feat_t = torch.from_numpy(feat).to(
                    device=device, dtype=frozen_tail.dtype)
                thw_t = torch.tensor(
                    val_grid_thws[len(val_teacher_cache)], dtype=torch.long,
                    device=device).unsqueeze(0)
                with torch.no_grad():
                    out = frozen_tail.forward_nograd(feat_t, thw_t)
                val_teacher_cache.append(out.cpu().float().numpy())
                del feat_t, out
            torch.cuda.empty_cache()

            opq_val_dlref = evaluate_opq_dlref_varlen(
                val_features, val_grid_thws, val_teacher_cache,
                frozen_tail, codebooks_std, R_std, args.embedding_dim,
                args.norm_mode, device)
            codec_val_dlref = evaluate_dlref_varlen(
                val_features, val_grid_thws, val_teacher_cache,
                frozen_tail, codec, args.norm_mode, device)
            delta_dlref = codec_val_dlref - opq_val_dlref
            print(f"  * OPQ   ΔL_ref/token = {opq_val_dlref:.4f}")
            print(f"  * Codec ΔL_ref/token = {codec_val_dlref:.4f}"
                  f"  (delta={delta_dlref:+.4f})")
            del val_teacher_cache

        if frozen_tail is not None:
            frozen_tail.to('cpu')
            del frozen_tail
            torch.cuda.empty_cache()

        # Save checkpoint
        codec_dir = os.path.join(ORFC_ROOT, 'checkpoints', args.backbone)
        os.makedirs(codec_dir, exist_ok=True)
        bt_tag = f"bt{bt_dim}" if bt_dim > 0 else "noBt"
        ws_tag = "ws" if args.warm_start_opq else "km"
        rate_tag = f"_lmbda{args.lmbda}" if args.lmbda > 0 else ""
        tau_tag = f"_tau{args.tau_start}" if args.tau_start > 0 else ""
        ckpt_name = (f"{args.layer}_K{args.K}_emb{args.embedding_dim}"
                     f"_{bt_tag}_{ws_tag}{rate_tag}{tau_tag}"
                     f"_lr{args.lr}_ep{args.epochs}_n{args.max_train_images}_s{args.seed}")
        ckpt_path = os.path.join(codec_dir, f"{ckpt_name}.pt")
        save_codec(codec, ckpt_path)
        print(f"  Codec saved: {ckpt_path}")

    # ================================================================
    #   (C) Evaluation on MMBench features (MSE / Rate / Acc)
    # ================================================================
    mmbench_dir = (Path(args.feat_root) / args.mmbench_subset
                   / args.backbone / args.layer)
    print(f"\n{'=' * 60}")
    print(f"  [MMBench] Evaluation (MSE / Rate / Acc)")
    print(f"  features: {mmbench_dir}")
    print(f"{'=' * 60}")

    opq_test_mse = None
    codec_test_mse = None
    rate_info = None

    if not mmbench_dir.exists():
        print(f"  [ERROR] MMBench features not found: {mmbench_dir}")
    else:
        mmbench_features, mmbench_basenames, _ = preload_features_varlen(
            mmbench_dir, max_images=args.max_mmbench_images, seed=args.seed)
        token_counts_mb = [f.shape[0] for f in mmbench_features]
        print(f"  Loaded {len(mmbench_features)} features, "
              f"T={min(token_counts_mb)}~{max(token_counts_mb)} "
              f"(mean={np.mean(token_counts_mb):.0f})")

        # --- OPQ MSE on MMBench ---
        opq_total_sq = 0.0
        opq_total_tokens = 0
        R_t_opq = torch.from_numpy(R_std).float().to(device)
        cb_t_opq = torch.from_numpy(np.stack(codebooks_std)).float().to(device)
        for feat in mmbench_features:
            X = torch.from_numpy(feat).float().unsqueeze(0).to(device)
            with torch.no_grad():
                Y, Mu, Std = batch_normalize_gpu(X, mode=args.norm_mode)
                C = feat.shape[1]
                flat = Y.reshape(-1, C) @ R_t_opq
                z_3d = flat.reshape(-1, opq_groups, args.embedding_dim) \
                       .permute(1, 0, 2).contiguous()
                z_hat_3d, _ = batched_assign(z_3d, cb_t_opq, device=device)
                flat_hat = z_hat_3d.permute(1, 0, 2).reshape(-1, C)
                Y_hat = (flat_hat @ R_t_opq.T).reshape(1, -1, C)
                opq_total_sq += ((Y - Y_hat) ** 2).sum().item()
                opq_total_tokens += feat.shape[0]
            del X, Y, Mu, Std, flat, z_3d, z_hat_3d, flat_hat, Y_hat
        del R_t_opq, cb_t_opq
        opq_test_mse = opq_total_sq / max(opq_total_tokens, 1)
        results['opq_mse'] = float(opq_test_mse)
        print(f"  * OPQ MSE/token = {opq_test_mse:.6f}")
        torch.cuda.empty_cache()

        # --- Codec MSE on MMBench ---
        codec_test_mse = evaluate_mse_varlen(
            mmbench_features, codec, args.norm_mode, device)
        results['codec_mse'] = float(codec_test_mse)
        delta_mse = codec_test_mse - opq_test_mse
        print(f"  * Codec MSE/token = {codec_test_mse:.6f} "
              f"(delta={delta_mse:+.6f})")

        # --- Rate on MMBench ---
        print(f"\n  Computing rate on MMBench features...")
        train_labels = _codec_labels_varlen(
            features_train, codec, args.norm_mode, device)
        train_pmf = _histogram_pmf(train_labels, num_groups, args.K)
        rate_info = evaluate_rate(
            mmbench_features, codec, args.norm_mode, device,
            train_pmf=train_pmf)
        results['rate_info'] = rate_info
        xent_str = f"xent={rate_info['xent_rate_bpt']:.2f}"
        if 'xent_train_bpt' in rate_info:
            xent_str += f"  xent_train={rate_info['xent_train_bpt']:.2f}"
        rans_str = ""
        if 'rans_bpt' in rate_info:
            rans_str = f"  rANS={rate_info['rans_bpt']:.2f}"
        if 'rans_train_bpt' in rate_info:
            rans_str += f"  rANS_train={rate_info['rans_train_bpt']:.2f}"
        print(f"  * Rate: {xent_str}  "
              f"H_emp={rate_info['empirical_entropy_bpt']:.2f}  "
              f"max={rate_info['max_rate_bpt']:.0f}{rans_str}  bits/token")

        # --- Acc on MMBench (replay, expensive) ---
        if args.eval_mmbench:
            mmbench_sample_list = args.mmbench_sample_list
            if (args.max_mmbench_images > 0
                    and len(mmbench_basenames) < 1000
                    and mmbench_sample_list):
                subset_list_dir = os.path.join(ORFC_ROOT, 'tmp_decoded')
                os.makedirs(subset_list_dir, exist_ok=True)
                subset_list_path = os.path.join(
                    subset_list_dir,
                    f"mmbench_subset_{len(mmbench_basenames)}.txt")
                with open(subset_list_path, 'w') as f:
                    for bn in mmbench_basenames:
                        f.write(f"{bn}\n")
                mmbench_sample_list = subset_list_path
                print(f"  Subset sample list ({len(mmbench_basenames)} entries):"
                      f" {subset_list_path}")

            decoded_feats = codec_encode_decode_varlen(
                mmbench_features, codec, args.norm_mode, device)
            suffix = getattr(args, 'result_suffix', '').strip()
            dir_tag = f"{args.layer}_K{args.K}_emb{args.embedding_dim}"
            if suffix:
                dir_tag += f"_{suffix}"
            elif args.ckpt_path:
                dir_tag = Path(args.ckpt_path).stem
            decoded_dir = os.path.join(
                ORFC_ROOT, 'tmp_decoded', args.backbone, dir_tag)
            os.makedirs(decoded_dir, exist_ok=True)
            for feat, bn in zip(decoded_feats, mmbench_basenames):
                np.save(os.path.join(decoded_dir, f"{bn}.npy"), feat)
            print(f"  Codec decoded -> {decoded_dir}")

            opq_decoded_dir = os.path.join(
                ORFC_ROOT, 'tmp_decoded', args.backbone,
                f"{dir_tag}_OPQ")
            os.makedirs(opq_decoded_dir, exist_ok=True)
            opq_decoded = pq_encode_decode_varlen(
                mmbench_features, codebooks_std, args.embedding_dim,
                args.norm_mode, device, R=R_std)
            for feat, bn in zip(opq_decoded, mmbench_basenames):
                np.save(os.path.join(opq_decoded_dir, f"{bn}.npy"), feat)
            print(f"  OPQ decoded   -> {opq_decoded_dir}")

            # Free GPU memory before launching replay subprocess
            codec.cpu()
            del decoded_feats, opq_decoded
            torch.cuda.empty_cache()

            orig_sample_list = args.mmbench_sample_list
            args.mmbench_sample_list = mmbench_sample_list

            acc_opq = evaluate_mmbench(opq_decoded_dir, args)
            acc_codec = evaluate_mmbench(decoded_dir, args)

            args.mmbench_sample_list = orig_sample_list

            if acc_opq is not None:
                results['mmbench_opq_acc'] = float(acc_opq)
                print(f"  * OPQ MMBench Acc = {acc_opq:.4f}")
            if acc_codec is not None:
                results['mmbench_codec_acc'] = float(acc_codec)
                print(f"  * Codec MMBench Acc = {acc_codec:.4f}")
            if acc_opq is not None and acc_codec is not None:
                delta = acc_codec - acc_opq
                results['mmbench_delta_acc'] = float(delta)
                print(f"  * Delta(Acc) = {delta:+.4f}")

        del mmbench_features

    # ================================================================
    #   Summary & save
    # ================================================================
    final_ppl = history[-1]['perplexity'] if history else 0.0
    final_dead = history[-1]['dead'] if history else 0
    results['final_ppl'] = float(final_ppl)
    results['final_dead'] = int(final_dead)
    if opq_val_dlref is not None:
        results['opq_val_dlref'] = float(opq_val_dlref)
    if codec_val_dlref is not None:
        results['codec_val_dlref'] = float(codec_val_dlref)

    print(f"\n{'=' * 60}")
    print(f"  Summary: {args.layer} K={args.K} emb={args.embedding_dim}"
          f"  bt={bt_dim if bt_dim > 0 else 'none'}")
    print(f"  bits/token: {bits_per_token:.0f} (G={num_groups})")
    print(f"  ppl={final_ppl:.1f}  dead={final_dead}/{num_groups * args.K}")
    if opq_val_dlref is not None and codec_val_dlref is not None:
        print(f"  OPQ   ΔL_ref/token = {opq_val_dlref:.4f}")
        print(f"  Codec ΔL_ref/token = {codec_val_dlref:.4f}"
              f"  delta={codec_val_dlref - opq_val_dlref:+.4f}")
    if opq_test_mse is not None:
        print(f"  OPQ   MSE/token = {opq_test_mse:.6f}")
    if codec_test_mse is not None:
        print(f"  Codec MSE/token = {codec_test_mse:.6f}"
              f"  delta={codec_test_mse - opq_test_mse:+.6f}")
    if rate_info is not None:
        ri = rate_info
        rans_s = f"  rANS={ri['rans_bpt']:.2f}" if 'rans_bpt' in ri else ""
        print(f"  Rate: xent={ri['xent_rate_bpt']:.2f}  "
              f"H_emp={ri['empirical_entropy_bpt']:.2f}  "
              f"max={ri['max_rate_bpt']:.0f}{rans_s} bits/token")
    if 'mmbench_opq_acc' in results:
        print(f"  MMBench OPQ  Acc = {results['mmbench_opq_acc']:.4f}")
    if 'mmbench_codec_acc' in results:
        print(f"  MMBench Codec Acc = {results['mmbench_codec_acc']:.4f}")
        if 'mmbench_opq_acc' in results:
            print(f"  Delta(Acc) = "
                  f"{results['mmbench_codec_acc'] - results['mmbench_opq_acc']:+.4f}")
    print(f"  Train time: std={std_time:.1f}s  codec={train_time:.1f}s")
    print(f"{'=' * 60}")

    results['train_time_std'] = float(std_time)
    results['train_time_spq'] = float(train_time)
    results['history'] = history

    out_dir = os.path.join(ORFC_ROOT, 'results', 'soft_pq', args.backbone)
    os.makedirs(out_dir, exist_ok=True)
    bt_tag = f"bt{bt_dim}" if bt_dim > 0 else "noBt"
    ws_tag = "ws" if args.warm_start_opq else "km"
    rate_tag = f"_lmbda{args.lmbda}" if args.lmbda > 0 else ""
    tau_tag = f"_tau{args.tau_start}" if args.tau_start > 0 else ""
    tag = (f"{args.layer}_K{args.K}_emb{args.embedding_dim}"
           f"_{bt_tag}_{ws_tag}{rate_tag}{tau_tag}"
           f"_lr{args.lr}_ep{args.epochs}_n{args.max_train_images}_s{args.seed}")
    result_suffix = getattr(args, 'result_suffix', '').strip()
    if result_suffix:
        safe = ''.join(c if (c.isalnum() or c in '-_') else '_' for c in result_suffix)
        tag = f"{tag}_{safe}"
    out_path = os.path.join(out_dir, f'{tag}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")

    return results


# ================================================================
#                    CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Soft-PQ for Qwen3-VL-4B variable-length visual features",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--layer", type=str, default="blk05")
    parser.add_argument("--K", type=int, default=64)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--norm_mode", type=str, default="per_image")
    parser.add_argument("--loss", type=str, default="delta_l_ref",
                        choices=["mse", "delta_l_ref"],
                        help="Training loss: delta_l_ref (default) or mse")
    parser.add_argument("--tail_dtype", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"],
                        help="dtype for FrozenTail blocks (delta_l_ref mode)")
    parser.add_argument("--grad_checkpoint", action="store_true", default=True,
                        help="Gradient checkpointing for FrozenTail (default on, use --no_grad_checkpoint to disable)")
    parser.add_argument("--no_grad_checkpoint", dest="grad_checkpoint",
                        action="store_false")
    parser.add_argument("--ckpt_segments", type=int, default=3,
                        help="Segment-level checkpointing: group tail into N segments (default 3, 0=per-block)")

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)

    parser.add_argument("--bottleneck_dim", type=int, default=1024,
                        help="Transform bottleneck dim (0=no transform, D=same-dim OrthogonalTransform)")
    parser.add_argument("--warm_start_opq", action="store_true",
                        help="Warm-start transform+codebooks from OPQ solution")
    parser.add_argument("--freeze_transform", action="store_true")
    parser.add_argument("--freeze_codebooks", action="store_true")
    parser.add_argument("--lmbda", type=float, default=0.0,
                        help="R-D Lagrange multiplier (0=no rate)")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--prior_floor", type=float, default=0.0)
    parser.add_argument("--tau_start", type=float, default=0.5)
    parser.add_argument("--tau_end", type=float, default=0.005)
    parser.add_argument("--tau_schedule", type=str, default="exponential",
                        choices=["exponential", "linear"])

    parser.add_argument("--max_train_images", type=int, default=5000)
    parser.add_argument("--kmeans_max_samples", type=int, default=2_000_000)
    parser.add_argument("--n_val", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size (images per step; smaller due to variable length)")

    parser.add_argument("--feat_root", type=str,
                        default=os.path.join(PROJECT_ROOT, "features"))
    parser.add_argument("--train_subset", type=str, default="train")
    parser.add_argument("--backbone", type=str, default="qwen3vl_4b")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--result_suffix", type=str, default="")

    # MMBench evaluation
    parser.add_argument("--eval_mmbench", action="store_true",
                        help="Run MMBench replay for Acc (MSE/Rate always computed)")
    parser.add_argument("--max_mmbench_images", type=int, default=0,
                        help="Max MMBench images (0=all, e.g. 100 for tuning)")
    parser.add_argument("--mmbench_subset", type=str,
                        default="mmbench_en_val_1000",
                        help="MMBench features subdir under feat_root (before /backbone/layer)")
    parser.add_argument("--mmbench_sample_list", type=str,
                        default=os.path.join(PROJECT_ROOT, "utils", "mmbench_en_val_1000.txt"),
                        help="Sample list for MMBench evaluation")
    parser.add_argument("--mmbench_data_dir", type=str,
                        default=os.path.join(PROJECT_ROOT, "data", "MMBench"),
                        help="MMBench dataset directory for replay")
    parser.add_argument("--model_path", type=str,
                        default="/data4/workspace/zlt/cache/hf/hub/Qwen3-VL-4B-Thinking",
                        help="Qwen3-VL model path for replay evaluation")
    parser.add_argument("--vit_layer", type=int, default=5,
                        help="ViT layer index for replay (must match --layer)")
    parser.add_argument("--eval_gpu", type=int, default=0,
                        help="GPU for MMBench replay evaluation")
    parser.add_argument("--python_exec", type=str,
                        default="/home/user/anaconda3/envs/qwen3vl_codec/bin/python")

    args = parser.parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()

"""Segmentation support for GAPC (VOC2012 mIoU + DEFLATE bpfp).

GAPC is a **raw-latent** codec (no normalisation, no retraining — paper
Algorithm 1 / §V).  The existing ``SegmentationEvaluator`` in
``ORFC/coding/orfc/backbone/wrapper.py`` already handles slide inference,
seg-head loading and mIoU accumulation, but ``CodecSegmentationEvaluator``
in ``run_soft_pq.py`` forces per-token-LN / per-image normalisation, which
breaks GAPC's design contract.  This module provides:

    * :class:`GAPCSegmentationEvaluator`
        Thin subclass that overrides ``quantize_tokens`` to pass raw
        ``[1, 1+N, D]`` tokens directly through the GAPC codec.

    * :func:`preload_seg_data`
        Cache image list / crops / GT / slide features **once**, so that
        multi-point sweeps do not redo preprocessing per operating point.

    * :func:`load_seg_backbone_and_head`
        Build a DINOv2 backbone + seg head on the target device **once**,
        then reuse across all operating points (critical for R-D sweeps).

    * :func:`evaluate_gapc_seg_point`
        End-to-end evaluation of a single ``(θ, quant_bits)`` GAPC operating
        point over all preloaded images — returns mIoU, per-class IoU, and
        aggregated DEFLATE bpfp (per-slide basis).

Per-slide rate
--------------
Each seg ``.npy`` has shape ``[num_slides, 1+N, D]`` (DINOv2-L/14 with
CROP_SIZE=512, patch=14 ⇒ 1+37×37=1370 tokens).  GAPC is applied
*independently per slide* (one mask per slide, matching Algorithm 1's
"per image" semantics, but here the "image" is a crop).  The reported
``bpfp`` is ``Σ bits / (num_slides × T × D)``, consistent with how
``run_soft_pq.py`` flattens slides for its rate curves.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_GAPC_DIR = os.path.dirname(os.path.abspath(__file__))
_ORFC_DIR = os.path.normpath(os.path.join(_GAPC_DIR, "..", "orfc"))
if _ORFC_DIR not in sys.path:
    sys.path.insert(0, _ORFC_DIR)

from backbone.wrapper import (                                  # noqa: E402
    SegmentationEvaluator,
    _DINOV2_REGISTRY,
    load_seg_head,
)

from gapc_codec import (                                         # noqa: E402
    GAPCCodec,
    RateRecord,
    aggregate_rate,
    measure_rate,
)


# ================================================================
#                    Evaluator (single-shot API)
# ================================================================


class GAPCSegmentationEvaluator(SegmentationEvaluator):
    """Segmentation evaluator for GAPC — raw-latent (no normalisation).

    Drop-in replacement for ``CodecSegmentationEvaluator`` (Soft-PQ) that
    keeps the downstream slide-inference / mIoU machinery intact while
    bypassing the per-token-LN / per-image normalisation step.
    """

    def __init__(self, codec: GAPCCodec, layer_idx: int,
                 voc_root: str, weights_root: str,
                 device: str = "cuda", feat_dim: int = 1024,
                 model_name: str = "dinov2_vitl14"):
        self.codec = codec
        self.layer_idx = layer_idx
        self.voc_root = voc_root
        self.weights_root = weights_root
        self.device = device
        self.feat_dim = feat_dim
        self.model_name = model_name
        self.norm_mode = "raw"                # documentation-only field

    @torch.no_grad()
    def quantize_tokens(self, tokens_np: np.ndarray) -> torch.Tensor:
        """GAPC on one slide's raw tokens ``[1+N, D]`` — no normalisation."""
        X = torch.from_numpy(tokens_np).float().unsqueeze(0).to(self.device)
        X_hat, _ = self.codec(X)
        return X_hat.squeeze(0)


# ================================================================
#         Preloading (shared across all operating points)
# ================================================================


def _load_seg_meta(image_list: str, voc_root: str, model_name: str):
    """Build mmseg test pipeline and load the image names list.

    Returns ``(test_pipeline, val_list)``.  Heavy imports are localised so
    that pure-codec unit tests do not pull in mmcv/mmseg.
    """
    import mmcv                                                  # noqa: F401
    from mmcv.parallel import collate                            # noqa: F401
    from mmseg.datasets.pipelines import Compose                 # noqa: F401

    reg = _DINOV2_REGISTRY[model_name]
    cfg = mmcv.Config.fromfile(reg["config"])
    cfg.data_root = voc_root

    class _LoadImage:
        """Pipeline stub that consumes an in-memory ``img`` dict."""
        def __call__(self, results):
            results["filename"] = results["ori_filename"] = None
            img = results["img"]
            results["img_shape"] = img.shape
            results["ori_shape"] = img.shape
            return results

    test_pipeline = Compose([_LoadImage()] + cfg.data.test.pipeline[1:])

    if not os.path.exists(image_list):
        raise FileNotFoundError(f"seg image list not found: {image_list}")
    with open(image_list, "r") as f:
        val_list = [ln.strip() for ln in f if ln.strip()]
    return test_pipeline, val_list


def preload_seg_data(seg_feat_dir: str, image_list: str, voc_root: str,
                     model_name: str, max_images: int = 0,
                     verbose: bool = True) -> List[dict]:
    """Cache features + crops + GT for all seg images *once*.

    Each record is a dict with keys:

        ``name``        str                     basename (no extension)
        ``features``    np.ndarray fp32         [num_slides, 1+N, D]
        ``crops``       list[(y1,x1,y2,x2)]     slide crop boxes
        ``gt``          np.ndarray uint8        [ori_h, ori_w]
        ``img_shape``   (h_img, w_img)          after mmseg test pipeline
    """
    from mmcv.parallel import collate
    from PIL import Image

    test_pipeline, val_list = _load_seg_meta(image_list, voc_root, model_name)

    # Pre-filter: only keep basenames that have BOTH a feature .npy and a
    # SegmentationClass .png.  voc2012_5000 mixes seg-labelled trainval
    # images with det-only train images, so a naive [:max_images] slice
    # loses most of the seg-labelled ones.  We filter first, then cap.
    kept_basenames = []
    drop_no_feat = 0
    drop_no_gt = 0
    drop_no_img = 0
    for name in val_list:
        if not os.path.exists(os.path.join(seg_feat_dir, f"{name}.npy")):
            drop_no_feat += 1
            continue
        if not os.path.exists(
            os.path.join(voc_root, "SegmentationClass", f"{name}.png")
        ):
            drop_no_gt += 1
            continue
        if not os.path.exists(
            os.path.join(voc_root, "JPEGImages", f"{name}.jpg")
        ):
            drop_no_img += 1
            continue
        kept_basenames.append(name)

    if max_images > 0 and len(kept_basenames) > max_images:
        kept_basenames = kept_basenames[:max_images]

    records: List[dict] = []
    slide_mismatch = 0
    for name in kept_basenames:
        feat_path = os.path.join(seg_feat_dir, f"{name}.npy")
        img_path = os.path.join(voc_root, "JPEGImages", f"{name}.jpg")
        gt_path = os.path.join(voc_root, "SegmentationClass", f"{name}.png")

        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)[:, :, ::-1]                       # RGB→BGR
        data = test_pipeline(dict(img=img_np))
        data = collate([data], samples_per_gpu=1)
        h_img = int(data["img"][0].shape[2])
        w_img = int(data["img"][0].shape[3])

        crops = SegmentationEvaluator.get_slide_crops(
            h_img, w_img,
            SegmentationEvaluator.CROP_SIZE,
            SegmentationEvaluator.STRIDE,
        )

        features = np.load(feat_path).astype(np.float32)
        if features.shape[0] != len(crops):
            if verbose:
                print(f"  [WARN] slide count mismatch for {name}: "
                      f"feat={features.shape[0]} crops={len(crops)} — skipped")
            slide_mismatch += 1
            continue

        gt = np.array(Image.open(gt_path))

        records.append({
            "name": name,
            "features": features,
            "crops": crops,
            "gt": gt,
            "img_shape": (h_img, w_img),
        })

    if verbose:
        print(f"  [seg] preloaded {len(records)} images "
              f"(from {len(val_list)} basenames; "
              f"no-feat={drop_no_feat} no-gt={drop_no_gt} "
              f"no-img={drop_no_img} slide-mismatch={slide_mismatch})")
    return records


def load_seg_backbone_and_head(model_name: str, weights_root: str,
                               device: str = "cuda"):
    """Load DINOv2 backbone + VOC seg head **once** for a sweep."""
    from dinov2.models import vision_transformer as vits

    reg = _DINOV2_REGISTRY[model_name]
    vit_builder = getattr(vits, reg["vit_fn"])
    backbone = vit_builder(**reg["vit_kwargs"])
    backbone_ckpt = os.path.join(weights_root, reg["pretrain"])
    backbone.load_state_dict(
        torch.load(backbone_ckpt, map_location="cpu"), strict=True
    )
    backbone = backbone.to(device).eval()

    head_ckpt = os.path.join(weights_root, reg["seg_head"])
    head = load_seg_head(
        head_ckpt, in_channels=reg["embed_dim"],
        num_classes=SegmentationEvaluator.NUM_CLASSES,
        device=device,
    )
    return backbone, head


# ================================================================
#       Per-operating-point evaluation (mIoU + bpfp together)
# ================================================================


@torch.no_grad()
def evaluate_gapc_seg_point(
    codec: GAPCCodec,
    seg_data: List[dict],
    backbone,
    head,
    layer_idx: int,
    device: str = "cuda",
    feat_dim: int = 1024,
    zip_level: int = 6,
    count_mask: bool = True,
    zip_workers: int = 8,
    verbose: bool = False,
) -> Tuple[dict, List[RateRecord]]:
    """Evaluate mIoU and bpfp for **one** GAPC operating point.

    Pipeline per image:
        1. Upload ``[num_slides, 1+N, D]`` to GPU, batched.
        2. Run GAPC once on the entire stack -> masks + ``X_hat``.
        3. Continue DINOv2 forward through the tail blocks + seg head for
           each slide (via ``slide_inference_decode``).
        4. Accumulate per-image class-confusion histogram.
        5. In parallel, dispatch CPU ``measure_rate`` over slides through
           a thread-pool (zlib releases the GIL).

    Returns
    -------
    results : dict with ``miou``, ``acc``, ``rate`` (agg).  Class-wise IoU
        is intentionally not returned — only the overall mIoU is reported.
    records : flat list of ``RateRecord`` (length = Σ num_slides).
    """
    codec.eval()
    NUM = SegmentationEvaluator.NUM_CLASSES
    IGN = SegmentationEvaluator.IGNORE_INDEX

    # Utility evaluator instance — we only use its slide_inference_decode.
    dummy = GAPCSegmentationEvaluator(
        codec=codec, layer_idx=layer_idx,
        voc_root="", weights_root="", device=device,
        feat_dim=feat_dim,
    )

    hist = np.zeros((NUM, NUM), dtype=np.int64)
    records: List[RateRecord] = []

    iterator = seg_data
    if verbose:
        try:
            from tqdm import tqdm
            iterator = tqdm(seg_data, desc="Seg eval", leave=False)
        except Exception:
            pass

    with ThreadPoolExecutor(max_workers=zip_workers) as ex:
        for item in iterator:
            features = item["features"]                          # [S, T, D]
            crops = item["crops"]
            gt = item["gt"]
            img_shape = item["img_shape"]

            X = torch.from_numpy(features).to(device, non_blocking=True)
            X_hat, _ = codec(X)                                  # [S, T, D]
            mask_gpu = codec._last_mask                          # [S, 1, D]
            mask_np = mask_gpu.squeeze(-2).cpu().numpy()         # [S, D]

            # --- CPU-threaded zlib rate per slide ---
            futs = [
                ex.submit(
                    measure_rate, features[s], mask_np[s],
                    zip_level, count_mask, int(codec.quant_bits),
                )
                for s in range(features.shape[0])
            ]

            # --- slide inference with the reconstructed X_hat ---
            quantized_list = [
                X_hat[s].unsqueeze(0) for s in range(X_hat.shape[0])
            ]
            preds_logits = dummy.slide_inference_decode(
                backbone, head, quantized_list, crops, img_shape
            )

            # --- upsample logits to GT size, argmax ---
            ori_h, ori_w = gt.shape[:2]
            if (img_shape[0], img_shape[1]) != (ori_h, ori_w):
                preds_logits = F.interpolate(
                    preds_logits, size=(ori_h, ori_w),
                    mode="bilinear", align_corners=False,
                )
            seg_pred = preds_logits.argmax(dim=1).squeeze(0).cpu().numpy()

            # --- accumulate class-confusion histogram (ignore 255) ---
            mask_valid = gt != IGN
            labels = gt[mask_valid].astype(np.int64)
            preds = seg_pred[mask_valid].astype(np.int64)
            hist += np.bincount(
                labels * NUM + preds, minlength=NUM * NUM
            ).reshape(NUM, NUM)

            records.extend(f.result() for f in futs)

    # ---- mIoU ----
    tp = np.diag(hist).astype(np.float64)
    denom = (hist.sum(axis=0) + hist.sum(axis=1) - tp).astype(np.float64)
    denom_safe = np.where(denom > 0, denom, 1.0)
    ious = tp / denom_safe
    valid = hist.sum(axis=1) > 0
    miou = float(ious[valid].mean()) if valid.any() else 0.0
    acc = float(tp.sum() / max(hist.sum(), 1))

    agg = aggregate_rate(records)
    return (
        {
            "miou": miou,
            "acc": acc,
            "rate": agg,
        },
        records,
    )

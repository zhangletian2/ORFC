"""Shared paths / constants for DINOv3 ViT-L/16 ORFC task eval."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
ORFC_ROOT = EVAL_DIR.parent
FEATCODEC_ROOT = ORFC_ROOT.parents[2]  # featcodec/
COFAI_ROOT = FEATCODEC_ROOT / "CoFAI"
FEAT_ROOT = FEATCODEC_ROOT / "features"

if str(ORFC_ROOT) not in sys.path:
    sys.path.insert(0, str(ORFC_ROOT))
if str(COFAI_ROOT) not in sys.path:
    sys.path.insert(0, str(COFAI_ROOT))
os.environ.setdefault("PROJECT_ROOT", str(COFAI_ROOT))

LAYERS = ("blk05", "blk10", "blk15", "blk20")
N_PREFIX = 5
PATCH = 16
EMBED_DIM = 1024
IMAGENET_TOKEN_HW = (14, 14)
IMAGENET_T = 1 + 4 + 14 * 14  # 201

BACKBONE_CKPT = str(
    COFAI_ROOT / "weights/dinov3/backbone/dinov3_vitl16_pretrain_lvd1689m.safetensors"
)
SEMSEG_HEAD = str(
    COFAI_ROOT / "weights/dinov3/semseg_head/dinov3_vitl16_semseg_ade20k_linear_head.pth"
)
DEPTH_HEAD = str(
    COFAI_ROOT / "weights/dinov3/dpt_head/dinov3_vitl16_depth_nyuv2_linear_head.pth"
)
CKPT_DIR = ORFC_ROOT / "checkpoints" / "dinov3_vitl16"
PROBE_DIR = EVAL_DIR / "probes" / "dinov3_vitl16"
RESULTS_DIR = EVAL_DIR / "results" / "dinov3_vitl16"
LOG_DIR = EVAL_DIR / "logs" / "dinov3_eval"

RAEV2_ROOT = FEATCODEC_ROOT / "third_party" / "RAEv2"
RAEV2_SRC = RAEV2_ROOT / "src"
RAEV2_DECODER_CFG = RAEV2_ROOT / "configs" / "decoder" / "ViTXL"
RAEV2_CKPT_DIR = RAEV2_ROOT / "pretrained_models" / "stage1" / "imagenet" / "dinov3l-k1"
RAEV2_DECODER_PT = RAEV2_CKPT_DIR / "decoder.pt"
RAEV2_STATS_PT = RAEV2_CKPT_DIR / "stats.pt"
RAEV2_RESULTS_DIR = EVAL_DIR / "results" / "dinov3_vitl16_raev2"
RAEV2_LOG_DIR = EVAL_DIR / "logs" / "dinov3_raev2"
RAEV2_IMG_SIZE = 256
RAEV2_TOKEN_HW = (16, 16)
RAEV2_T = 1 + 4 + 16 * 16  # 261

IMAGENET_ROOT = FEATCODEC_ROOT / "data/imagenet/images/val"
IMAGENET_TEST_LIST = FEATCODEC_ROOT / "utils/imagenet_selected_pathname500.txt"
IMAGENET_TEST_LABELS = FEATCODEC_ROOT / "utils/imagenet_selected_label500.txt"
IMAGENET_TRAIN_LIST = FEATCODEC_ROOT / "utils/imagenet_selected_pathname5000.txt"
IMAGENET_TRAIN_LABELS = FEATCODEC_ROOT / "utils/imagenet_selected_label5000.txt"
IMAGENET_TRAIN_LIST_STAGE2 = (
    FEATCODEC_ROOT / "utils/imagenet_selected_pathname5000_stage2.txt"
)
IMAGENET_TRAIN_FEAT = FEAT_ROOT / "train/dinov3_vitl16"
IMAGENET_TRAIN_FEAT_STAGE2 = FEAT_ROOT / "train/dinov3_vitl16_stage2"
IMAGENET_TEST_FEAT = FEAT_ROOT / "test/dinov3_vitl16"

ADE_ROOT = COFAI_ROOT / "data/ADEChallengeData2016"
ADE_IMG_DIR = ADE_ROOT / "images/validation"
ADE_ANN_DIR = ADE_ROOT / "annotations/validation"
ADE_FEAT_ROOT = FEAT_ROOT / "ade20k_val/dinov3_vitl16"

NYU_ROOT = COFAI_ROOT / "data/NYU"
NYU_LIST = NYU_ROOT / "nyu_test.txt"
NYU_FEAT_ROOT = FEAT_ROOT / "nyu_depth/dinov3_vitl16"

NORM_MODE = "split_reg_cls_patch"

_CKPT_RE = re.compile(
    r"^(blk\d+)_K(\d+)_emb(\d+)_bt\d+_ws_lmbda([0-9.]+)_"
)


def layer_idx(layer: str) -> int:
    return int(layer.replace("blk", ""))


def decode_slot(layer: str) -> int:
    """Dinov3TimmBackbone.slot: curr_layer = slot - 1 = layer_idx."""
    return layer_idx(layer) + 1


def list_orfc_ckpts(ckpt_dir: Path | None = None) -> list[Path]:
    d = Path(ckpt_dir or CKPT_DIR)
    return sorted(d.glob("blk*_split_reg_cls_patch_*.pt"))


def parse_ckpt_name(path: Path | str) -> dict:
    name = Path(path).name
    if name.endswith(".pt"):
        name = name[:-3]
    m = _CKPT_RE.match(name)
    info = {"stem": name, "layer": None, "K": None, "emb": None, "lmbda": None}
    if m:
        info.update(
            {
                "layer": m.group(1),
                "K": int(m.group(2)),
                "emb": int(m.group(3)),
                "lmbda": float(m.group(4)),
            }
        )
    else:
        for lyr in LAYERS:
            if name.startswith(lyr + "_"):
                info["layer"] = lyr
                break
    return info


def load_label_map(path: Path) -> dict[str, int]:
    out = {}
    with open(path) as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) >= 2:
                out[parts[0]] = int(parts[1])
    return out


def nyu_stem_from_rel(img_rel: str) -> str:
    """bathroom/rgb_00045.jpg -> bathroom__rgb_00045"""
    return img_rel.replace("/", "__").replace(".jpg", "")


def resolve_nyu_rgb(img_rel: str) -> Path:
    cand = NYU_ROOT / "test" / img_rel
    if cand.is_file():
        return cand
    cand = NYU_ROOT / img_rel
    if cand.is_file():
        return cand
    raise FileNotFoundError(f"NYU RGB not found: {img_rel}")


def resolve_nyu_depth(depth_rel: str) -> Path:
    cand = NYU_ROOT / "test" / depth_rel
    if cand.is_file():
        return cand
    cand = NYU_ROOT / depth_rel
    if cand.is_file():
        return cand
    raise FileNotFoundError(f"NYU depth not found: {depth_rel}")

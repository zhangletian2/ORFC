"""RAEv2 ViTXL load + differentiable decode (orfcv2 residual L1 KD)."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

ORFCV2 = Path(__file__).resolve().parent
ORFC = ORFCV2.parent / "orfc"
if str(ORFC / "eval") not in sys.path:
    sys.path.insert(0, str(ORFC / "eval"))

from dinov3_eval_common import (  # noqa: E402
    RAEV2_DECODER_CFG,
    RAEV2_DECODER_PT,
    RAEV2_SRC,
    RAEV2_STATS_PT,
    RAEV2_TOKEN_HW,
)

if str(RAEV2_SRC) not in sys.path:
    sys.path.insert(0, str(RAEV2_SRC))
from stage1.decoders.decoder import GeneralDecoder  # noqa: E402
from stage1.decoders.utils import ViTMAEConfig  # noqa: E402
from stage1.rae import _load_normalization_stats  # noqa: E402


def load_raev2_vitxl(num_patches: int | None = None) -> nn.Module:
    """Load ViTXL without AutoConfig (config.json has patch_size='SHOULD BE RELOADED')."""
    if num_patches is None:
        num_patches = RAEV2_TOKEN_HW[0] * RAEV2_TOKEN_HW[1]
    cfg_path = Path(RAEV2_DECODER_CFG) / "config.json"
    raw = json.loads(cfg_path.read_text())
    config = ViTMAEConfig(
        hidden_size=1024,
        patch_size=16,
        image_size=int(16 * math.sqrt(num_patches)),
        decoder_hidden_size=int(raw["decoder_hidden_size"]),
        decoder_intermediate_size=int(raw["decoder_intermediate_size"]),
        decoder_num_attention_heads=int(raw["decoder_num_attention_heads"]),
        decoder_num_hidden_layers=int(raw["decoder_num_hidden_layers"]),
        hidden_act=raw.get("hidden_act", "gelu"),
        hidden_dropout_prob=float(raw.get("hidden_dropout_prob", 0.0)),
        attention_probs_dropout_prob=float(raw.get("attention_probs_dropout_prob", 0.0)),
        layer_norm_eps=float(raw.get("layer_norm_eps", 1e-12)),
        initializer_range=float(raw.get("initializer_range", 0.02)),
        num_channels=int(raw.get("num_channels", 3)),
        qkv_bias=bool(raw.get("qkv_bias", True)),
        num_attention_heads=int(raw.get("num_attention_heads", 12)),
        num_hidden_layers=int(raw.get("num_hidden_layers", 12)),
        intermediate_size=int(raw.get("intermediate_size", 3072)),
    )
    decoder = GeneralDecoder(config, num_patches=num_patches)
    print(f"Loading pretrained decoder from {RAEV2_DECODER_PT}")
    state = torch.load(RAEV2_DECODER_PT, map_location="cpu", weights_only=False)
    keys = decoder.load_state_dict(state, strict=False)
    if keys.missing_keys:
        print(f"  missing keys: {keys.missing_keys}")
    if keys.unexpected_keys:
        print(f"  unexpected keys: {len(keys.unexpected_keys)}")
    return decoder


def load_raev2_head(device):
    if not RAEV2_DECODER_PT.is_file() or not RAEV2_STATS_PT.is_file():
        raise FileNotFoundError(
            f"RAEv2 weights missing:\n  {RAEV2_DECODER_PT}\n  {RAEV2_STATS_PT}"
        )
    n_patches = RAEV2_TOKEN_HW[0] * RAEV2_TOKEN_HW[1]
    decoder = load_raev2_vitxl(n_patches).to(device).eval()
    for p in decoder.parameters():
        p.requires_grad_(False)
    mean, var, do_norm = _load_normalization_stats(str(RAEV2_STATS_PT))
    if mean is not None:
        mean = mean.to(device)
        var = var.to(device)
    print(
        f"  RAEv2 decoder={RAEV2_DECODER_PT.name}  "
        f"stats={tuple(mean.shape) if mean is not None else None}"
    )
    return decoder, mean, var, do_norm


def raev2_preprocess(img: Image.Image, size: int) -> torch.Tensor:
    """Center-crop to square then resize — RAEv2 `scripts/stage1/sample.py`."""
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    img = img.resize((size, size), Image.BICUBIC)
    arr = torch.from_numpy(np.asarray(img, dtype=np.float32)) / 255.0
    return arr.permute(2, 0, 1)


def decoder_forward(decoder, patches):
    """Differentiable ViTXL forward (frozen weights, no gradient checkpoint).

    Matches RAEv2 `GeneralDecoder.forward(..., drop_cls_token=False)` and
    `rae_tail/train.py` `decoder_forward(..., use_checkpoint=False)`.
    """
    x_ = decoder.decoder_embed(patches)
    x_ = decoder.interpolate_latent(x_)
    cls_tok = decoder.trainable_cls_token.expand(x_.shape[0], -1, -1)
    h = torch.cat([cls_tok, x_], dim=1)
    h = h + decoder.decoder_pos_embed
    for layer in decoder.decoder_layers:
        h = layer(h, None, False)
        if isinstance(h, tuple):
            h = h[0]
    h = decoder.decoder_norm(h)
    logits = decoder.decoder_pred(h)
    return logits[:, 1:, :]


def patches_to_pixels(decoder, patches, clamp=False):
    logits = decoder_forward(decoder, patches)
    rec = decoder.unpatchify(logits)
    if clamp:
        rec = rec.clamp(0.0, 1.0)
    return rec

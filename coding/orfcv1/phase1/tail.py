"""The frozen ViT tail and the cache/layer consistency assertion.

Both functions used to live in ``p1_fixed_rate.py``, which also carried the
whole v8 analysis chain (separable projection, ANOVA, remainder decomposition).
Plan v9 uses none of that, so the two pieces phase 1 actually needs are lifted
here and the rest is deleted.

``check_layer`` is the assertion that caught defect E1: three run scripts never
passed ``--layer 5`` while their teacher caches were built at block 5, so the
"distortion" being measured was the difference between two different networks
(``blocks[6:]`` against ``blocks[21:]``).  Nothing else in the pipeline notices,
which is why the check is mechanical and sits at every entry point.
"""

import re
from pathlib import Path

import torch


def build_tail(layer, device):
    """blocks[layer+1:] + final norm, on ``device``; everything else on CPU."""
    from backbone.wrapper import Dinov2Wrapper
    from soft_pq import FrozenTail

    wrapper = Dinov2Wrapper(
        head_layers=1, model_name="dinov2_vitl14", device=device)
    blocks = list(wrapper.backbone.blocks)
    for block in blocks[:layer + 1]:
        block.cpu()
    if wrapper.head is not None:
        wrapper.head.cpu()
    torch.cuda.empty_cache()
    return FrozenTail(blocks[layer + 1:], wrapper.backbone.norm, device=device)


def layer_from_cache_name(path):
    """Parse the blkNN tag out of a cache filename; None when absent."""
    match = re.search(r"_blk(\d+)_", Path(path).name)
    return int(match.group(1)) if match else None


def check_layer(layer, *paths):
    """Fail when ``layer`` disagrees with a cache's blkNN tag."""
    for path in paths:
        if path is None:
            continue
        tag = layer_from_cache_name(path)
        if tag is not None and tag != layer:
            raise ValueError(
                f"layer {layer} disagrees with cache tag blk{tag:02d} in "
                f"{Path(path).name}: the teacher tail is blocks[{tag + 1}:] "
                f"but the in-process tail would be blocks[{layer + 1}:]")

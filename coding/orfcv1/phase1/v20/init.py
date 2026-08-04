"""Build OPQ and all-mode k-means initialisations for V20."""

import argparse
import json

import torch

from .config import SPECS, activate
from ..v15.init import build


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=tuple(SPECS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    config, anchor = activate(args.profile)
    target = config.init_dir(anchor, "orfc_cayley") / "codec.pt"
    if target.exists() and not args.force:
        raise SystemExit(f"{target} exists; pass --force to replace")
    result = build(anchor, torch.device(args.device),
                   parameterization="orfc_cayley", config=config)
    result["plan"] = config.PLAN
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

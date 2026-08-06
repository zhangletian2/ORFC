"""Create the immutable V28 validation split manifest."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--select", type=int, default=500)
    args = parser.parse_args()
    source = Path(args.ids)
    lines = [line.strip() for line in source.read_text().splitlines() if line.strip()]
    if len(lines) != 3000 or len(set(lines)) != len(lines):
        raise SystemExit("expected 3000 unique validation IDs")
    order = np.random.default_rng(args.seed).permutation(len(lines))
    select, rest = order[:args.select], order[args.select:]
    payload = {
        "contract": "v28_fixed_validation_select",
        "seed": args.seed,
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "validation_select": {
            "count": len(select), "indices": select.tolist(),
            "image_ids": [lines[i] for i in select]},
        "validation_rest": {
            "count": len(rest), "indices": rest.tolist(),
            "image_ids": [lines[i] for i in rest]},
    }
    out = Path(args.out)
    if out.exists():
        old = json.loads(out.read_text())
        if old != payload:
            raise SystemExit("existing manifest differs; refusing overwrite")
        print(out)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(out)


if __name__ == "__main__":
    main()

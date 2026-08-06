"""Train and export the shared tree checkpoint after a PASS capacity gate."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from codec_v1 import load_codec_v1
from .. import engine, tail as tail_mod
from ..v21.config import SPECS, activate
from . import common, nested
from .capacity_gate import train_coverage


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity-gate", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    started = time.time(); device = torch.device(args.device)
    gate_path = Path(args.capacity_gate).resolve()
    gate = json.loads(gate_path.read_text())
    if gate.get("verdict") != "PASS":
        raise SystemExit("coverage export requires a PASS capacity gate")
    if gate.get("parameterization") != "parent_conditioned_residual_tree":
        raise SystemExit("capacity gate uses the wrong parameterization")
    block, anchor_name = gate["block"], gate["anchor"]
    if block not in SPECS:
        raise SystemExit("unknown block in capacity gate")
    config = activate(block); anchor = config.ANCHOR_BY_NAME[anchor_name]
    source_path = Path(gate["source_codec"])
    source = load_codec_v1(source_path, device=device)
    codec = nested.from_independent(source)
    bits = common.mode_bits(codec)
    if list(bits) != gate["mode_bits"]:
        raise SystemExit("source menu differs from the capacity gate")
    allocations = np.asarray(gate["allocations"], dtype=np.int64)
    if len(allocations) != 4 or any(
            common.nominal_rate(row, bits) != anchor.rate
            for row in allocations):
        raise SystemExit("capacity allocations violate the export contract")
    train_paths = config.load_split("train_fit")
    train_images = min(16, int(gate["train_images"])) if args.smoke \
        else int(gate["train_images"])
    epochs = min(2, int(gate["epochs"])) if args.smoke \
        else int(gate["epochs"])
    train = engine.ResidentSet(
        *train_paths[:2], train_paths[2][:train_images], device)
    tail = tail_mod.build_tail(config.LAYER, device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_coverage(
        codec, tail, train, allocations, epochs,
        int(gate["batch"]), float(gate["lr"]),
        float(gate["tau_schedule"][0]), float(gate["tau_schedule"][1]),
        int(args.seed))
    manifest = {
        "plan": "v30_capacity_pass_coverage_export",
        "capacity_gate": str(gate_path), "capacity_gate_verdict": "PASS",
        "block": block, "anchor": anchor_name,
        "source_codec": str(source_path.resolve()),
        "allocations": allocations.tolist(), "seed": int(args.seed),
        "smoke": bool(args.smoke),
        "epochs": epochs, "batch": int(gate["batch"]),
        "train_images": train.count,
        "peak_memory_bytes": (int(torch.cuda.max_memory_allocated(device))
                              if device.type == "cuda" else 0),
        "seconds": time.time() - started}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    nested.save_checkpoint(
        codec, source_path, out / "nested_codec.pt",
        extra={"capacity_gate": gate, "coverage_export": manifest})
    (out / "coverage_export.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

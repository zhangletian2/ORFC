"""V33 delivery verification — 1-opt, same-graph ORFC Tail MSE, ranking, rate.

Acceptance hooks from the two-stage plan (no full 100-epoch ORFC retrain here):

1. Exhaustive two-group-transfer 1-opt certificate on the delivery allocation
   (reuses :func:`phase1.v33.search.one_opt_certificate`).
2. Strict same-graph Tail MSE vs an ORFC reference on the fixed ``train_val``
   list (default 500 images; same resident for both codecs).
3. Top-k ranking-consistency hook: search order vs full-retrain scores
   (scores supplied by the caller; retrain itself is out of scope).
4. Nominal rate == ``anchor.rate``; rANS / downstream are CLI stubs with
   documented call sites (no fake implementation).
5. Pipeline order via ``--print-pipeline`` / README.

Smoke
-----
``python -m phase1.v33.verify --smoke --out /tmp/verify.json`` exercises the
report writer with synthetic 1-opt / ranking / rate checks (no feature cache).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .. import engine, tail as tail_mod
from ..v12 import config as V12
from ..v21.config import SPECS, activate
from . import checkpoint as ckpt
from . import distortion as D
from . import ranking
from . import search
from . import valset

# Defaults match v11 ORFC pool / N14 matched recipe (blk20).
DEFAULT_ORFC_DIR = Path(
    "/data4/workspace/zlt/featcodec/ORFC/coding/orfc/checkpoints/dinov2_vitl14")
DEFAULT_ORFC_STEM = {
    "R64": ("blk20_K4_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003"
            "_ep100_n5000_s42"),
    "R96": ("blk20_K8_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003"
            "_ep100_n5000_s42"),
}

# Documented external hooks (not implemented in v33).
RANS_TODO = {
    "status": "TODO",
    "note": (
        "V33 has no in-tree rANS encoder for nested multi-mode labels. "
        "ORFC real-rate path lives under CoFAI examples and ORFC calibrator; "
        "phase1 downstream accuracy/seg is separate from Tail MSE verify."
    ),
    "call_sites": [
        "featcodec/CoFAI/examples/orfc_2446/dinov2/offline/test_cls.py"
        " :: rans_encode_per_image / _rans_encode_bpt",
        "featcodec/ORFC/coding/orfcv1/phase1/v12/eval_downstream.py"
        " :: ImageNet-500 / VOC (accuracy & mIoU, not rANS)",
        "featcodec/ORFC/coding/orfcv1/phase1/v31/downstream.py"
        " :: nested-codec downstream hook",
    ],
}

PIPELINE_COMMANDS = """
# V33 recommended order (cwd: featcodec/ORFC/coding/orfcv1)
# 1) Phase-1 strict-fair supernet
python -m phase1.v33.train --phase 1 --anchor R64 --run-id <RUN>

# 2) Noise floor on a late phase-1 checkpoint
python -m phase1.v33.noise_floor \\
  --checkpoint phase1/v33/<RUN>/R64/checkpoint.pt \\
  --out phase1/v33/<RUN>/R64/noise.json --anchor R64

# 3) Switch-point probe (ordered checkpoints; need past peak + both stable)
python -m phase1.v33.switch_probe \\
  --checkpoints <ckpt_a> <ckpt_b> <ckpt_c> \\
  --noise-floor-json phase1/v33/<RUN>/R64/noise.json \\
  --out phase1/v33/<RUN>/R64/probe.json

# 4) Local search → allocation.npy
python -m phase1.v33.search \\
  --checkpoint <switch_ckpt.pt> \\
  --out phase1/v33/<RUN>/R64/search.json --anchor R64

# 5) Phase-2 frozen-U0 specialisation
python -m phase1.v33.train --phase 2 --anchor R64 --run-id <RUN>_p2 \\
  --source phase1/v33/<RUN>/R64/checkpoint.pt \\
  --allocation phase1/v33/<RUN>/R64/allocation.npy

# 6) Delivery verify (1-opt + same-graph ORFC + rate; optional ranking JSON)
python -m phase1.v33.verify \\
  --checkpoint phase1/v33/<RUN>_p2/R64/checkpoint.pt \\
  --allocation phase1/v33/<RUN>/R64/allocation.npy \\
  --orfc-ref <stem_or_path> \\
  --out phase1/v33/<RUN>_p2/R64/verify.json
""".strip()


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj))


def resolve_allocation(allocation_arg, payload=None, checkpoint_path=None,
                       groups=32):
    """Load allocation from CLI path, checkpoint meta, or sibling ``allocation.npy``."""
    if allocation_arg is not None:
        return ckpt.load_allocation(allocation_arg, groups=groups)
    if payload is not None:
        values = ckpt.meta_allocation(payload)
        if values is not None:
            return ckpt.load_allocation(values, groups=groups)
    if checkpoint_path is not None:
        sibling = Path(checkpoint_path).with_name("allocation.npy")
        if sibling.exists():
            return ckpt.load_allocation(sibling, groups=groups)
    raise ValueError(
        "allocation required: pass --allocation, embed in checkpoint meta, "
        "or place allocation.npy next to the checkpoint")


def check_nominal_rate(allocation, mode_bits, rate):
    actual = ckpt.allocation_rate(allocation, mode_bits)
    ok = int(actual) == int(rate)
    return {
        "passed": ok,
        "nominal_rate": int(actual),
        "anchor_rate": int(rate),
        "mode_bits": list(map(int, mode_bits)),
        "allocation": [int(x) for x in torch.as_tensor(allocation).reshape(-1)],
    }


def check_one_opt(score_fn, allocation, bits):
    """Exhaustive neighbourhood 1-opt (``propose_top_k=None``)."""
    return search.one_opt_certificate(
        score_fn, allocation, bits, propose_top_k=None)


def check_topk_ranking_consistency(search_scores, retrain_scores,
                                   ids=None, tie_threshold=0.0):
    """Compare search ranking vs full-retrain scores (lower is better).

    Does **not** run retrain; callers supply both score vectors in the same
    candidate order.  Reports argsort agreement, sparse Kendall-τ, and whether
    the top-1 identity matches.
    """
    search_scores = np.asarray(search_scores, dtype=np.float64).reshape(-1)
    retrain_scores = np.asarray(retrain_scores, dtype=np.float64).reshape(-1)
    if search_scores.shape != retrain_scores.shape:
        raise ValueError(
            f"score length mismatch: search {search_scores.shape} vs "
            f"retrain {retrain_scores.shape}")
    n = int(search_scores.size)
    if ids is None:
        ids = [str(i) for i in range(n)]
    else:
        ids = [str(x) for x in ids]
        if len(ids) != n:
            raise ValueError("ids length must match scores")

    order_search = np.argsort(search_scores, kind="stable")
    order_retrain = np.argsort(retrain_scores, kind="stable")
    order_match = bool(np.array_equal(order_search, order_retrain))
    top1_match = bool(order_search[0] == order_retrain[0]) if n else True
    tau, conc, disc, decisive, dropped = ranking.sparse_kendall_tau(
        search_scores, retrain_scores, tie_threshold=tie_threshold)
    return {
        "passed": order_match,
        "n": n,
        "ids": ids,
        "search_scores": search_scores.tolist(),
        "retrain_scores": retrain_scores.tolist(),
        "search_order": [ids[i] for i in order_search.tolist()],
        "retrain_order": [ids[i] for i in order_retrain.tolist()],
        "argsort_identical": order_match,
        "top1_identical": top1_match,
        "sparse_kendall_tau": None if tau is None else float(tau),
        "n_concordant": int(conc),
        "n_discordant": int(disc),
        "n_decisive": int(decisive),
        "n_dropped": int(dropped),
        "tie_threshold": float(tie_threshold),
        "note": (
            "Full top-k retrain is out of band; this hook only compares "
            "provided score vectors."),
    }


def load_score_vector(path):
    """JSON ``[s...]`` or ``{\"scores\": [...], \"ids\": [...]}``."""
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, list):
        return payload, None
    if isinstance(payload, dict):
        scores = payload.get("scores", payload.get("search_scores"))
        if scores is None:
            raise ValueError(f"{path}: need 'scores' list")
        return scores, payload.get("ids")
    raise ValueError(f"unsupported ranking JSON in {path}")


def default_orfc_ref(anchor_name, orfc_dir=None):
    stem = DEFAULT_ORFC_STEM.get(str(anchor_name))
    if stem is None:
        return None
    root = Path(orfc_dir) if orfc_dir is not None else DEFAULT_ORFC_DIR
    return root / f"{stem}.pt"


def resolve_orfc_paths(orfc_ref, orfc_dir=None, anchor_name="R64"):
    """Return ``(stem, pt_path, npz_path)`` or raise with a clear message.

    ``orfc_ref`` may be empty (use default stem), a stem, a ``.pt`` / ``.npz``
    path, or a directory containing the default stem files.
    """
    root = Path(orfc_dir) if orfc_dir is not None else DEFAULT_ORFC_DIR
    if orfc_ref is None or str(orfc_ref).strip() == "":
        default = default_orfc_ref(anchor_name, root)
        if default is None:
            raise FileNotFoundError(
                f"no default ORFC stem for anchor {anchor_name}; "
                f"pass --orfc-ref explicitly")
        orfc_ref = default

    path = Path(orfc_ref)
    if path.suffix in {".pt", ".npz"}:
        stem = path.stem
        parent = path.parent
        pt = parent / f"{stem}.pt"
        npz = parent / f"{stem}.npz"
    elif path.is_dir():
        default = default_orfc_ref(anchor_name, path)
        if default is None or not default.exists():
            raise FileNotFoundError(
                f"{path} has no default ORFC .pt for {anchor_name}")
        stem = default.stem
        pt, npz = default, path / f"{stem}.npz"
    else:
        # Treat as stem under orfc_dir.
        stem = path.name if path.suffix == "" else str(orfc_ref)
        pt = root / f"{stem}.pt"
        npz = root / f"{stem}.npz"

    if not pt.exists():
        raise FileNotFoundError(
            f"ORFC checkpoint missing: {pt}\n"
            f"Pass --orfc-ref STEM|.pt|.npz or --orfc-dir DIR. "
            f"Default for {anchor_name}: {default_orfc_ref(anchor_name)}")
    if not npz.exists():
        raise FileNotFoundError(
            f"ORFC sidecar missing: {npz} (required with {pt})")
    return stem, pt, npz


def load_orfc_multimode(anchor, stem, pt_path, npz_path, device):
    """Wrap ORFC ``.npz``/``.pt`` as a multimode codec for ``engine`` eval.

    Prefer the registered v11 loader when both files sit in
    ``v11.config.ORFC_CHECKPOINT_DIR`` (full cross-checks).  Otherwise build
    the same FeatureCodecV1 wrapper from the explicit paths.
    """
    from ..v11 import config as V11C
    from ..v11 import orfc_baseline

    pt_path, npz_path = Path(pt_path), Path(npz_path)
    registered = Path(V11C.ORFC_CHECKPOINT_DIR).resolve()
    if (pt_path.resolve().parent == registered
            and npz_path.resolve().parent == registered):
        codec, _, _, provenance = orfc_baseline.load_orfc_codec(
            anchor, stem, device)
        return codec, provenance
    return _load_orfc_from_paths(anchor, stem, pt_path, npz_path, device)


def _load_orfc_from_paths(anchor, stem, pt_path, npz_path, device):
    """Minimal ORFC → FeatureCodecV1 wrap (same layout as ``orfc_baseline``)."""
    from codec_v1 import FeatureCodecV1
    from cayley import DirectOrthogonalTransform
    from multimode_pq import MultiModeSoftPQ

    archive = np.load(npz_path, allow_pickle=False)
    rotation = np.ascontiguousarray(archive["R"], dtype=np.float32)
    codebooks = np.ascontiguousarray(archive["codebooks"], dtype=np.float32)
    dim = V12.GROUPS * V12.DIM
    k = int(anchor.mode_sizes[anchor.uniform_mode])
    if rotation.shape != (dim, dim):
        raise SystemExit(
            f"ORFC R shape {rotation.shape} != {(dim, dim)} for {stem}")
    if codebooks.shape != (V12.GROUPS, k, V12.DIM):
        raise SystemExit(
            f"ORFC codebooks shape {codebooks.shape} != "
            f"{(V12.GROUPS, k, V12.DIM)} for {stem}")

    blob = torch.load(pt_path, map_location="cpu", weights_only=False)
    state = blob.get("state_dict", blob) if isinstance(blob, dict) else blob
    book_pt = state.get("pq.codebooks")
    if book_pt is not None:
        book_pt = book_pt.detach().cpu().numpy().astype(np.float32)
        if not np.array_equal(book_pt, codebooks):
            raise SystemExit(
                f"ORFC {stem}: codebooks in .npz and .pt disagree")

    pq = MultiModeSoftPQ(V12.GROUPS, anchor.mode_sizes, V12.DIM).to(device)
    for mode, quantizer in enumerate(pq.quantizers):
        if mode == anchor.uniform_mode:
            quantizer.codebooks.data.copy_(
                torch.from_numpy(codebooks).to(device))
        else:
            quantizer.codebooks.data.zero_()
        if hasattr(quantizer, "log_prior"):
            quantizer.log_prior.data.zero_()

    transform = DirectOrthogonalTransform(dim).to(device)
    transform.rotation.data.copy_(torch.from_numpy(rotation).to(device))
    codec = FeatureCodecV1(pq, transform).to(device).eval()
    rot = codec.transform.get_rotation()
    orth = float((rot.t() @ rot - torch.eye(dim, device=device)).norm())
    provenance = {
        "stem": stem, "npz": str(npz_path), "pt": str(pt_path),
        "uniform_K": k, "orthogonality_error": orth,
        "loader": "v33.verify._load_orfc_from_paths",
    }
    return codec, provenance


@torch.no_grad()
def evaluate_v33_mean(codec, tail, resident, allocation, image_batch=16):
    matrix = D.evaluate(
        codec, tail, resident, allocation, image_batch=image_batch)
    per_image = np.asarray(matrix, dtype=np.float64).reshape(-1)
    return float(per_image.mean()), per_image


@torch.no_grad()
def evaluate_orfc_mean(orfc_codec, tail, resident, allocation, image_batch=16):
    matrix = engine.evaluate_allocations(
        orfc_codec, tail, resident, np.asarray(allocation)[None, :],
        image_batch=image_batch, pair_budget=image_batch, per_image=True)
    per_image = np.asarray(matrix[0], dtype=np.float64)
    return float(per_image.mean()), per_image


def compare_same_graph(v33_mean, orfc_mean, v33_per_image=None,
                       orfc_per_image=None):
    delta = float(v33_mean - orfc_mean)
    rel = delta / max(abs(orfc_mean), 1e-12)
    payload = {
        "v33_tail_mse": float(v33_mean),
        "orfc_tail_mse": float(orfc_mean),
        "delta_v33_minus_orfc": delta,
        "relative_delta": rel,
        "v33_better": bool(v33_mean < orfc_mean),
        "passed": bool(v33_mean < orfc_mean),
        "split": "train_val",
        "note": (
            "Same ResidentSet / tail / MSE definition for both codecs. "
            "ORFC scored at its uniform allocation; V33 at the delivery "
            "allocation."),
    }
    if v33_per_image is not None and orfc_per_image is not None:
        diff = np.asarray(v33_per_image) - np.asarray(orfc_per_image)
        payload["per_image_delta_mean"] = float(diff.mean())
        payload["per_image_delta_std"] = float(diff.std())
        payload["n_images"] = int(diff.size)
    return payload


def rans_and_downstream_hooks(rans_cmd=None, downstream_cmd=None):
    payload = dict(RANS_TODO)
    payload["rans_cli"] = rans_cmd
    payload["downstream_cli"] = downstream_cmd
    if rans_cmd:
        payload["rans_note"] = (
            "Caller supplied --rans-cmd; verify does not execute it.")
    if downstream_cmd:
        payload["downstream_note"] = (
            "Caller supplied --downstream-cmd; verify does not execute it. "
            "Suggested: python -m phase1.v12.eval_downstream …")
    return payload


def run_verify(*, checkpoint, allocation_arg=None, anchor_name="R64",
               block="blk20", device="cuda", n_images=None, image_batch=16,
               orfc_ref=None, orfc_dir=None, skip_orfc=False, skip_one_opt=False,
               search_ranking=None, retrain_scores=None, tie_threshold=0.0,
               rans_cmd=None, downstream_cmd=None):
    started = time.time()
    activate(block)
    engine.configure_precision(V12.ALLOW_TF32)
    device = torch.device(device)
    anchor = V12.ANCHOR_BY_NAME[anchor_name]
    n_images = valset.N_VAL if n_images is None else int(n_images)

    codec, payload = ckpt.load_checkpoint(checkpoint, device=device)
    bits = tuple(codec.pq.mode_bits)
    if bits != tuple(anchor.mode_bits):
        raise SystemExit(
            f"codec mode_bits {bits} != anchor {anchor.mode_bits}")
    allocation = resolve_allocation(
        allocation_arg, payload=payload, checkpoint_path=checkpoint,
        groups=codec.pq.G)

    rate_report = check_nominal_rate(allocation, bits, anchor.rate)
    if not rate_report["passed"]:
        raise SystemExit(
            f"INVALID_EXPERIMENT: nominal rate "
            f"{rate_report['nominal_rate']} != {anchor.rate}")

    tail = tail_mod.build_tail(V12.LAYER, device)
    resident = valset.load_val_resident(
        device, n_images=n_images, image_batch=image_batch)

    one_opt = None
    if not skip_one_opt:
        score_fn = search.make_codec_scorer(
            codec, tail, resident, image_batch=image_batch)
        one_opt = check_one_opt(score_fn, allocation, bits)

    v33_mean, v33_imgs = evaluate_v33_mean(
        codec, tail, resident, allocation, image_batch=image_batch)

    orfc_report = {"skipped": True}
    if not skip_orfc:
        stem, pt, npz = resolve_orfc_paths(
            orfc_ref, orfc_dir=orfc_dir, anchor_name=anchor.name)
        orfc_codec, provenance = load_orfc_multimode(
            anchor, stem, pt, npz, device)
        uniform = engine.uniform_allocation(anchor)
        orfc_mean, orfc_imgs = evaluate_orfc_mean(
            orfc_codec, tail, resident, uniform, image_batch=image_batch)
        orfc_report = compare_same_graph(
            v33_mean, orfc_mean, v33_imgs, orfc_imgs)
        orfc_report["skipped"] = False
        orfc_report["orfc_stem"] = stem
        orfc_report["orfc_pt"] = str(pt)
        orfc_report["orfc_npz"] = str(npz)
        orfc_report["orfc_provenance"] = {
            "orthogonality_error": provenance.get("orthogonality_error"),
            "uniform_K": provenance.get("uniform_K"),
            "loader": provenance.get("loader", "v11.orfc_baseline"),
        }
        orfc_report["v33_n_images"] = int(len(v33_imgs))

    ranking_report = None
    if search_ranking is not None or retrain_scores is not None:
        if search_ranking is None or retrain_scores is None:
            raise SystemExit(
                "ranking check needs both --search-ranking and --retrain-scores")
        s_scores, s_ids = load_score_vector(search_ranking)
        r_scores, r_ids = load_score_vector(retrain_scores)
        ids = s_ids if s_ids is not None else r_ids
        ranking_report = check_topk_ranking_consistency(
            s_scores, r_scores, ids=ids, tie_threshold=tie_threshold)

    report = {
        "plan": "v33_delivery_verify",
        "smoke": False,
        "checkpoint": str(Path(checkpoint).resolve()),
        "anchor": anchor.name,
        "block": block,
        "rate": anchor.rate,
        "n_images": n_images,
        "image_batch": image_batch,
        "allocation": [int(x) for x in torch.as_tensor(allocation).reshape(-1)],
        "nominal_rate": rate_report,
        "one_opt": one_opt,
        "v33_tail_mse": float(v33_mean),
        "vs_orfc": orfc_report,
        "topk_ranking": ranking_report,
        "rans_downstream": rans_and_downstream_hooks(rans_cmd, downstream_cmd),
        "meta": payload.get("meta") or {},
        "seconds": time.time() - started,
    }
    # Gates: rate + optional 1-opt / ORFC / ranking (top1 or full argsort).
    gates = {
        "nominal_rate": rate_report["passed"],
        "one_opt": None if one_opt is None else bool(one_opt.get("is_1opt")),
        "vs_orfc": None if orfc_report.get("skipped") else bool(
            orfc_report.get("v33_better")),
        "topk_ranking": None if ranking_report is None else bool(
            ranking_report["argsort_identical"]
            or ranking_report["top1_identical"]),
    }
    report["gates"] = gates
    report["passed"] = all(v for v in gates.values() if v is not None)
    return report


def run_smoke(seed=0):
    """Synthetic checks only — no feature cache / ORFC / GPU required."""
    started = time.time()
    groups, bits, rate = 8, (1, 2, 3), 16
    score, _ = search.synthetic_score_fn(bits, rate, seed=seed)(groups)
    # Climb to a local optimum so the 1-opt gate exercises a true certificate.
    climbed = search.run_search(
        score, score, groups, bits, rate,
        eval_budget=400, propose_top_k=20, max_steps=16,
        n_random_starts=2, topk=2, seed=seed, simplified=False,
        certify=True)
    allocation = np.asarray(climbed["winner"]["allocation"], dtype=np.int64)
    rate_report = check_nominal_rate(allocation, bits, rate)
    one_opt = climbed["certificate"]
    if one_opt is None:
        one_opt = check_one_opt(score, allocation, bits)

    # Planted ranking: same order on both sides.
    search_scores = [3.0, 1.0, 2.5, 2.0]
    retrain_scores = [2.95, 1.05, 2.4, 2.1]
    ranking_ok = check_topk_ranking_consistency(
        search_scores, retrain_scores, ids=["a", "b", "c", "d"])

    # Discordant smoke sample for the API (not gated).
    ranking_bad = check_topk_ranking_consistency(
        [1.0, 2.0, 3.0], [3.0, 2.0, 1.0], ids=["x", "y", "z"])

    report = {
        "plan": "v33_delivery_verify",
        "smoke": True,
        "geometry": {"groups": groups, "mode_bits": list(bits), "rate": rate},
        "allocation": allocation.tolist(),
        "nominal_rate": rate_report,
        "one_opt": one_opt,
        "vs_orfc": {
            "skipped": True,
            "reason": "smoke uses synthetic scores; pass --checkpoint for "
                      "strict same-graph ORFC Tail MSE",
            "default_orfc_ref": str(default_orfc_ref("R64")),
        },
        "topk_ranking": ranking_ok,
        "topk_ranking_discordant_example": ranking_bad,
        "rans_downstream": rans_and_downstream_hooks(),
        "pipeline": PIPELINE_COMMANDS,
        "seconds": time.time() - started,
    }
    report["gates"] = {
        "nominal_rate": rate_report["passed"],
        "one_opt": bool(one_opt.get("is_1opt")),
        "vs_orfc": None,
        "topk_ranking": bool(ranking_ok["argsort_identical"]),
    }
    report["passed"] = all(
        v for v in report["gates"].values() if v is not None)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--print-pipeline", action="store_true",
                        help="print recommended command order and exit")
    parser.add_argument("--checkpoint", default=None,
                        help=f"{ckpt.FORMAT} delivery checkpoint")
    parser.add_argument("--allocation", default=None,
                        help="delivery allocation (.npy / .json); else meta "
                             "or sibling allocation.npy")
    parser.add_argument("--out", required=False, default=None)
    parser.add_argument("--block", default="blk20", choices=tuple(SPECS))
    parser.add_argument("--anchor", default="R64",
                        choices=list(V12.ANCHOR_BY_NAME))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-images", type=int, default=None,
                        help=f"val images (default {valset.N_VAL})")
    parser.add_argument("--image-batch", type=int, default=16)
    parser.add_argument(
        "--orfc-ref", default=None,
        help="ORFC stem, .pt/.npz path, or empty for default matched stem "
             f"(R64 → {DEFAULT_ORFC_STEM['R64']})")
    parser.add_argument(
        "--orfc-dir", default=str(DEFAULT_ORFC_DIR),
        help="directory of ORFC .pt/.npz pairs (default: coding/orfc/checkpoints/…)")
    parser.add_argument("--skip-orfc", action="store_true")
    parser.add_argument("--skip-one-opt", action="store_true")
    parser.add_argument("--search-ranking", default=None,
                        help="JSON scores from search (same order as retrain)")
    parser.add_argument("--retrain-scores", default=None,
                        help="JSON scores from full retrain of the same top-k")
    parser.add_argument("--tie-threshold", type=float, default=0.0)
    parser.add_argument("--rans-cmd", default=None,
                        help="documented only; not executed")
    parser.add_argument("--downstream-cmd", default=None,
                        help="documented only; not executed")
    args = parser.parse_args(argv)

    if args.print_pipeline:
        print(PIPELINE_COMMANDS)
        return 0

    if args.smoke:
        report = run_smoke()
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint is required unless --smoke")
        report = run_verify(
            checkpoint=args.checkpoint,
            allocation_arg=args.allocation,
            anchor_name=args.anchor,
            block=args.block,
            device=args.device,
            n_images=args.n_images,
            image_batch=args.image_batch,
            orfc_ref=args.orfc_ref,
            orfc_dir=args.orfc_dir,
            skip_orfc=args.skip_orfc,
            skip_one_opt=args.skip_one_opt,
            search_ranking=args.search_ranking,
            retrain_scores=args.retrain_scores,
            tie_threshold=args.tie_threshold,
            rans_cmd=args.rans_cmd,
            downstream_cmd=args.downstream_cmd,
        )

    text = json.dumps(report, indent=2, default=_json_default)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
    print(text)
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())

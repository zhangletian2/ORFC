#!/usr/bin/env python
"""conv2 main/recon0 vs original ORFC RD plots (Pareto only) + BD-Rate.

One PNG per (layer, task, method): recon0 vs ORFC and main vs ORFC.
Style matches orfcv2/plot_shareR_rd.py.

    python -u plot_conv2_orfc_rd.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

HERE = Path(__file__).resolve().parent
ORFC = HERE.parent / "orfc"
UTILS = HERE.parents[1] / "utils"
for p in (str(UTILS),):
    if p not in sys.path:
        sys.path.insert(0, p)

from cal_bd_rate import BD_RATE  # noqa: E402

ORFC_COLOR = "#1f77b4"
OURS_COLOR = "#d62728"
ANCHOR_COLOR = "0.45"

LAYERS = ("blk05", "blk10", "blk15", "blk20")
METHODS = ("recon0", "main")
TASKS = ("seg", "depth")

VOC_UNQUANT_MIOU = 0.8165787
_VOC_ANCHOR_JSON = (
    HERE / "results" / "similarity_merge" / "dinov2_vitl14"
    / "voc100_blk05_blk10_blk15_blk20_f0.5-0.25.json"
)


def voc_unquant_miou():
    fp = _VOC_ANCHOR_JSON
    if not fp.is_file():
        return VOC_UNQUANT_MIOU
    layers = json.loads(fp.read_text()).get("layers") or {}
    vals = [float(row["anchor_miou"]) for row in layers.values()
            if row.get("anchor_miou") is not None]
    return float(sum(vals) / len(vals)) if vals else VOC_UNQUANT_MIOU


def pareto_front(points, minimize=False):
    """Keep undominated (K, rate, quality) points; quality monotone in rate."""
    kept = []
    if minimize:
        best_y = float("inf")
        for p in sorted(points, key=lambda t: (t[1], t[2])):
            y = p[2]
            if y < best_y:
                kept.append(p)
                best_y = y
    else:
        best_y = float("-inf")
        for p in sorted(points, key=lambda t: (t[1], -t[2])):
            y = p[2]
            if y > best_y:
                kept.append(p)
                best_y = y
    return kept


def _finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def _pt(K, bpfp, quality, bits=None):
    if not (_finite(bpfp) and _finite(quality) and bpfp > 0):
        return None
    row = {"K": int(K), "bpfp": float(bpfp), "quality": float(quality)}
    if _finite(bits):
        row["bits"] = float(bits)
    return row


def _from_block(K, block, task):
    if not block:
        return None
    timing = (block.get("timing") or {}).get(task) or {}
    if task == "seg":
        return _pt(K, timing.get("bpfp"), (block.get("seg") or {}).get("miou"),
                   timing.get("bits_per_image"))
    return _pt(K, timing.get("bpfp"), (block.get("depth") or {}).get("rmse"),
               timing.get("bits_per_image"))


def collect_task_jsons(result_dir: Path):
    """Parse per-job *_tasks.json into {layer: {method: {task: [rows]}}}."""
    data = {
        layer: {m: {"seg": [], "depth": []} for m in (*METHODS, "orfc")}
        for layer in LAYERS
    }
    anchors = {layer: None for layer in LAYERS}
    orfc_seen = {layer: set() for layer in LAYERS}

    def add(layer, method, task, row):
        if row is None or layer not in data:
            return
        k = row["K"]
        bucket = data[layer][method][task]
        for i, old in enumerate(bucket):
            if old["K"] == k:
                bucket[i] = row
                return
        bucket.append(row)

    for fp in sorted(result_dir.glob("*_tasks.json")):
        d = json.loads(fp.read_text())
        layer = d.get("layer")
        if layer not in data:
            continue
        K = int(d.get("K") or 0)
        if d.get("anchor_rmse") is not None:
            anchors[layer] = float(d["anchor_rmse"])
        stem = fp.stem
        if d.get("orfc_only") or (K == 2 and "conv2" not in stem):
            for task in TASKS:
                add(layer, "orfc", task, _from_block(K, d.get("orfc"), task))
            orfc_seen[layer].add(K)
            continue
        method = None
        if "conv2_main_" in stem or d.get("residual_ablation") == "main":
            method = "main"
        elif "conv2_recon0_" in stem or d.get("residual_ablation") == "recon0":
            method = "recon0"
        if method is None:
            continue
        for task in TASKS:
            add(layer, method, task, _from_block(K, d.get("bilinear"), task))
        if K not in orfc_seen[layer]:
            for task in TASKS:
                add(layer, "orfc", task, _from_block(K, d.get("orfc"), task))
            if any(_from_block(K, d.get("orfc"), t) is not None for t in TASKS):
                orfc_seen[layer].add(K)

    for layer in LAYERS:
        for method in data[layer]:
            for task in TASKS:
                data[layer][method][task].sort(key=lambda r: r["K"])
    return data, anchors


def rows_to_tuples(rows):
    return [(r["K"], r["bpfp"], r["quality"]) for r in rows]


def tuples_to_rows(pts, src_rows):
    by_k = {r["K"]: r for r in src_rows}
    out = []
    for k, bpfp, q in pts:
        row = dict(by_k.get(k) or {"K": k})
        row["bpfp"] = bpfp
        row["quality"] = q
        out.append(row)
    return out


def bd_rate_pct(orfc_pts, ours_pts, minimize):
    if len(orfc_pts) < 2 or len(ours_pts) < 2:
        return None
    r1 = [p[1] for p in orfc_pts]
    d1 = [p[2] for p in orfc_pts]
    r2 = [p[1] for p in ours_pts]
    d2 = [p[2] for p in ours_pts]
    val = BD_RATE(r1, d1, r2, d2, piecewise=1, higher_better=not minimize)
    if val is None or (isinstance(val, float) and not math.isfinite(val)):
        return None
    return float(val)


def build_summary(data, anchors):
    voc_h = voc_unquant_miou()
    out = {
        "title": "conv2 main/recon0 vs original ORFC (Pareto RD + BD-Rate)",
        "protocol": "voc100_nyu80_rans",
        "units": {
            "seg": "VOC2012-100 mIoU vs rANS bpfp = bits / (src_tokens * 1024)",
            "depth": "NYU test80 RMSE vs rANS bpfp",
            "bd_rate_pct": (
                "Bjontegaard delta rate vs ORFC, PCHIP on Pareto points; "
                "negative = fewer bits than ORFC at the same quality; "
                "null if quality ranges do not overlap"
            ),
        },
        "notes": {
            "main": "hat X = U(ORFC(E(X)))",
            "recon0": "hat X = U(ORFC(E(X))) + F_phi(X0, 0)",
            "orfc": "original FeatureCodec on all tokens; includes K=2",
            "pareto": "undominated points only; quality monotone in rate",
            "excluded": "linear4 not included; ORFC K=512 missing",
        },
        "unquantized": {"voc_miou": voc_h, "nyu_rmse": None},
        "layers": {},
    }
    nyu_vals = [v for v in anchors.values() if v is not None]
    if nyu_vals:
        out["unquantized"]["nyu_rmse"] = float(sum(nyu_vals) / len(nyu_vals))

    for layer in LAYERS:
        layer_out = {}
        for task in TASKS:
            minimize = (task == "depth")
            pts = {
                m: data[layer][m][task]
                for m in (*METHODS, "orfc")
            }
            pareto = {
                m: tuples_to_rows(
                    pareto_front(rows_to_tuples(pts[m]), minimize), pts[m])
                for m in pts
            }
            bd = {}
            orfc_t = rows_to_tuples(pareto["orfc"])
            for m in METHODS:
                bd[f"{m}_vs_orfc"] = bd_rate_pct(
                    orfc_t, rows_to_tuples(pareto[m]), minimize)
            layer_out[task] = {
                "points": pts,
                "pareto": pareto,
                "bd_rate_pct": bd,
            }
        out["layers"][layer] = layer_out
    return out


def _ylim(ys, cap=None, rmse=False):
    y_lo, y_hi = min(ys), max(ys)
    span = max(y_hi - y_lo, 1e-9)
    if rmse:
        pad = max(0.02, 0.10 * span)
        lo = math.floor((y_lo - pad) * 20 - 1e-9) / 20
        hi = math.ceil((y_hi + pad) * 20 + 1e-9) / 20
        if cap is not None:
            hi = min(hi, cap)
        if hi - lo < 0.10:
            mid = 0.5 * (y_lo + y_hi)
            lo, hi = mid - 0.05, mid + 0.05
        return lo, hi
    pad = min(0.04, max(0.008, 0.12 * span))
    lo = math.floor((y_lo - pad) * 100 - 1e-9) / 100
    hi = math.ceil((y_hi + pad) * 100 + 1e-9) / 100
    if cap is not None:
        hi = min(hi, cap)
    if hi - lo < 0.02:
        mid = round((y_lo + y_hi) / 2, 2)
        lo, hi = mid - 0.01, mid + 0.01
        if cap is not None:
            hi = min(hi, cap)
    return lo, hi


def _y_locator(lo, hi, rmse=False):
    span = hi - lo
    if rmse:
        if span <= 0.20:
            return MultipleLocator(0.05)
        if span <= 0.50:
            return MultipleLocator(0.10)
        return MultipleLocator(0.20)
    if span <= 0.04:
        step = 0.01
    elif span <= 0.10:
        step = 0.02
    elif span <= 0.25:
        step = 0.04
    else:
        step = 0.05
    return MultipleLocator(step)


def plot_rd(orfc, ours, out_path: Path, title, ylabel, ours_label,
            y_cap=None, hline=None, hline_label=None, legend_loc="lower right",
            rmse=False):
    fig, ax = plt.subplots(figsize=(4.8, 3.6), dpi=160)
    ox, oy = [p[1] for p in orfc], [p[2] for p in orfc]
    rx, ry = [p[1] for p in ours], [p[2] for p in ours]
    ys = oy + ry
    if hline is not None:
        ys = ys + [hline]
        ax.axhline(
            hline, color=ANCHOR_COLOR, linestyle=":", linewidth=1.0,
            zorder=2, label=hline_label or "anchor",
        )
    ax.plot(
        ox, oy, color=ORFC_COLOR, marker="o", markersize=7,
        markeredgecolor="white", markeredgewidth=0.6, linewidth=1.4,
        zorder=3, label="ORFC",
    )
    ax.plot(
        rx, ry, color=OURS_COLOR, marker="s", markersize=7,
        markeredgecolor="white", markeredgewidth=0.6, linewidth=1.4,
        zorder=3, label=ours_label,
    )
    ax.set_xlabel("rANS BPFP")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    y_lo, y_hi = _ylim(ys, cap=y_cap, rmse=rmse)
    ax.set_ylim(y_lo, y_hi)
    xmin, xmax = min(ox + rx), max(ox + rx)
    pad = 0.08 * (xmax - xmin) if xmax > xmin else 0.01
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_locator(_y_locator(y_lo, y_hi, rmse=rmse))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(frameon=False, loc=legend_loc)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def plot_one(layer, task, method, orfc, ours, out_dir, bd, anchors, voc_h):
    minimize = (task == "depth")
    ylabel = "depth RMSE" if minimize else "seg mIoU"
    title_task = "depth" if minimize else "segmentation"
    out = out_dir / f"{layer}_{task}_{method}_vs_orfc.png"
    hline = anchors.get(layer) if minimize else voc_h
    plot_rd(
        orfc, ours, out,
        title=f"{layer}  {title_task}",
        ylabel=ylabel,
        ours_label=method,
        y_cap=None if minimize else 1.0,
        hline=hline,
        hline_label="unquantized",
        legend_loc="upper right" if minimize else "lower right",
        rmse=minimize,
    )
    print(f"{layer} {task}  ORFC n={len(orfc)}  K={[p[0] for p in orfc]}")
    print(f"{layer} {task}  {method:6s} n={len(ours)}  K={[p[0] for p in ours]}")
    if bd is not None:
        print(f"  BD-Rate {method} vs ORFC = {bd:+.2f}%")
    else:
        print(f"  BD-Rate {method} vs ORFC = n/a (no quality overlap)")
    for name, pts in (("ORFC", orfc), (method, ours)):
        for k, x, y in pts:
            print(f"  {name:6s} K{k:<3d}  BPFP={x:.4f}  {ylabel.split()[-1]}={y:.4f}")
    print(out)
    return out


def dump_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--result_dir",
        default=str(HERE / "results" / "bilinear_orfc_jointopq" / "tasks"),
    )
    p.add_argument(
        "--json_out",
        default=str(
            HERE / "results" / "bilinear_orfc_jointopq" / "tasks"
            / "conv2_main_recon0_vs_orfc_rd.json"),
    )
    p.add_argument(
        "--out_dir",
        default=str(HERE / "results" / "bilinear_orfc_jointopq" / "plots"),
    )
    p.add_argument("--layers", nargs="+", default=list(LAYERS))
    p.add_argument("--methods", nargs="+", default=list(METHODS))
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument("--no_plot", action="store_true")
    args = p.parse_args()

    result_dir = Path(args.result_dir)
    data, anchors = collect_task_jsons(result_dir)
    summary = build_summary(data, anchors)
    dump_json(Path(args.json_out), summary)

    missing_k2 = [
        layer for layer in args.layers
        if not any(r["K"] == 2 for r in data[layer]["orfc"]["seg"])
    ]
    if missing_k2:
        print(f"WARNING: ORFC K=2 missing for {missing_k2} "
              f"(run eval_bilinear_orfc_joint_tasks.py --orfc_only --K 2)")

    if args.no_plot:
        return

    voc_h = summary["unquantized"]["voc_miou"]
    out_dir = Path(args.out_dir)
    for layer in args.layers:
        for task in args.tasks:
            block = summary["layers"][layer][task]
            orfc = rows_to_tuples(block["pareto"]["orfc"])
            for method in args.methods:
                ours = rows_to_tuples(block["pareto"][method])
                if not orfc or not ours:
                    print(f"skip {layer} {task} {method}: empty Pareto")
                    continue
                plot_one(
                    layer, task, method, orfc, ours, out_dir,
                    block["bd_rate_pct"][f"{method}_vs_orfc"],
                    anchors, voc_h,
                )


if __name__ == "__main__":
    main()

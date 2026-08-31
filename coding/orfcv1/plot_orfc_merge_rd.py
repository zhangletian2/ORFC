#!/usr/bin/env python
"""ORFC vs merge RD. One PNG per (layer, task).

Default (``--style full``): full-token ORFC, greedy/ToMe at
frac=0.25 (reuse R/PQ), greedy/ToMe at frac=0.50, ToMe + trained
R/PQ.  Classification Acc (higher better); depth NYU test80 RMSE
(lower better) with unquantized anchor; VOC mIoU.  Each series
keeps Pareto points.

``--style paper``: depth + segmentation only, ORFC vs greedy merge
(red squares) plus the unquantized original-feature hline.
blk05/blk10 use f=0.25; blk15/blk20 use f=0.5.  Written to
``plots/greedy_orfc/``.

``--style paper_tome``: same as paper, plus ToMe (green triangles)
at the same per-layer f.  Written to ``plots/greedy_tome/``.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

ORFC_COLOR = "#1f77b4"
MERGE_COLOR = "#d62728"
TOME_COLOR = "#2ca02c"
TRAINED_COLOR = "#ff7f0e"
ANCHOR_COLOR = "0.45"

# Collapsed ToMe retrain (blk20 K4 λ=0.5 Acc~20%).  Prefer λ=0 when both exist.
TRAINED_COLLAPSE_ACC = 0.5

LAYERS = ("blk05", "blk10", "blk15", "blk20")
TASKS = ("cls", "depth", "seg")
FRAC_025 = 0.25
FRAC_050 = 0.50
# Paper RD: shallower layers keep more tokens; deeper layers merge more.
PAPER_FRAC = {
    "blk05": FRAC_025,
    "blk10": FRAC_025,
    "blk15": FRAC_050,
    "blk20": FRAC_050,
}

# VOC2012-100 mIoU of unquantized DINOv2-L replay (remaining ViT + linear
# head). Same at every layer; from eval_similarity_merge_seg.py.
_VOC_ANCHOR_JSON = (
    Path(__file__).resolve().parent
    / "results" / "similarity_merge" / "dinov2_vitl14"
    / "voc100_blk05_blk10_blk15_blk20_f0.5-0.25.json"
)
VOC_UNQUANT_MIOU = 0.8165787


def pareto_front(points, minimize=False):
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


def _merge_frac(d, fp=None):
    c = d.get("config") or {}
    if c.get("merge_frac") is not None:
        return float(c["merge_frac"])
    if fp is not None:
        name = Path(fp).name
        for tag, val in (("_f0.5_", 0.5), ("_f0.25_", 0.25)):
            if tag in name:
                return val
    return FRAC_025


def _frac_close(a, b):
    return abs(float(a) - float(b)) < 1e-6


def _row_task_point(row, task, want_orfc=False):
    k = int(row["K"])
    ckpt = row.get("ckpt", "")
    if task == "cls":
        src = row.get("cls_orfc") if want_orfc else row.get("cls_merge")
        if not src:
            return None
        bpfp = src.get("bpfp") if want_orfc else src.get("bpfp_with_match_side")
        if bpfp is None or src.get("acc") is None:
            return None
        return (k, float(bpfp), float(src["acc"]), ckpt)
    if task == "depth":
        if want_orfc:
            bo = row.get("depth_orfc_rate") or {}
            if row.get("depth_orfc_rmse") is None or bo.get("bpfp") is None:
                return None
            return (k, float(bo["bpfp"]), float(row["depth_orfc_rmse"]), ckpt)
        bm = row.get("depth_merge_rate") or {}
        if row.get("depth_merge_rmse") is None or bm.get("bpfp_with_match_side") is None:
            return None
        return (k, float(bm["bpfp_with_match_side"]),
                float(row["depth_merge_rmse"]), ckpt)
    if task == "seg":
        src = row.get("seg_orfc") if want_orfc else row.get("seg_merge")
        if not src:
            return None
        bpfp = src.get("bpfp") if want_orfc else src.get("bpfp_with_match_side")
        if bpfp is None or src.get("miou") is None:
            return None
        return (k, float(bpfp), float(src["miou"]), ckpt)
    return None


def load_merge_points(result_dir: Path, layer: str, task: str):
    orfc, merge = [], []
    anchor = None
    minimize = (task == "depth")
    for fp in sorted(result_dir.glob(f"frozen_orfc_merge_{layer}_f*_shareR_*.json")):
        d = json.loads(fp.read_text())
        frac = _merge_frac(d, fp)
        if d.get("nyu_anchor_rmse") is not None:
            anchor = float(d["nyu_anchor_rmse"])
        for row in d.get("runs") or []:
            op = _row_task_point(row, task, want_orfc=True)
            if op is not None:
                orfc.append(op)
            if _frac_close(frac, FRAC_025):
                mp = _row_task_point(row, task, want_orfc=False)
                if mp is not None:
                    merge.append(mp)
    return pareto_front(orfc, minimize), pareto_front(merge, minimize), anchor


def load_greedy_frac_points(result_dir: Path, layer: str, task: str, frac: float):
    pts = []
    minimize = (task == "depth")
    for fp in sorted(result_dir.glob(f"frozen_orfc_merge_{layer}_f*_shareR_*.json")):
        d = json.loads(fp.read_text())
        if not _frac_close(_merge_frac(d, fp), frac):
            continue
        for row in d.get("runs") or []:
            p = _row_task_point(row, task, want_orfc=False)
            if p is not None:
                pts.append(p)
    return pareto_front(pts, minimize)


def load_voc_anchor(layer: str | None = None) -> float:
    fp = _VOC_ANCHOR_JSON
    if fp.is_file():
        layers = json.loads(fp.read_text()).get("layers") or {}
        if layer and layer in layers and layers[layer].get("anchor_miou") is not None:
            return float(layers[layer]["anchor_miou"])
        vals = [float(row["anchor_miou"]) for row in layers.values()
                if row.get("anchor_miou") is not None]
        if vals:
            return float(sum(vals) / len(vals))
    return VOC_UNQUANT_MIOU


def load_tome_points(result_dir: Path, layer: str, task: str, frac=FRAC_025):
    pts = []
    minimize = (task == "depth")
    for fp in sorted(result_dir.glob(f"frozen_tome_merge_{layer}_f*_shareR_*.json")):
        d = json.loads(fp.read_text())
        if not _frac_close(_merge_frac(d, fp), frac):
            continue
        for row in d.get("runs") or []:
            p = _row_task_point(row, task, want_orfc=False)
            if p is not None:
                pts.append(p)
    return pareto_front(pts, minimize)


def _pick_trained_runs(runs):
    """One run per K: drop collapsed Acc, prefer λ=0 over λ=0.5."""
    by_k = {}
    for row in runs:
        acc = (row.get("cls_trained") or {}).get("acc")
        if acc is not None and float(acc) < TRAINED_COLLAPSE_ACC:
            continue
        k = int(row["K"])
        prev = by_k.get(k)
        if prev is None:
            by_k[k] = row
            continue
        lam, plam = float(row.get("lmbda", 0.5)), float(prev.get("lmbda", 0.5))
        if lam == 0.0 and plam != 0.0:
            by_k[k] = row
        elif lam != 0.0 and plam == 0.0:
            continue
        elif acc is not None:
            pa = (prev.get("cls_trained") or {}).get("acc") or 0.0
            if acc > pa:
                by_k[k] = row
    return [by_k[k] for k in sorted(by_k)]


def load_trained_points(trained_dir: Path, layer: str, task: str):
    pts = []
    minimize = (task == "depth")
    fp = trained_dir / f"eval_tasks_{layer}_tome.json"
    if not fp.is_file():
        return []
    d = json.loads(fp.read_text())
    for row in _pick_trained_runs(d.get("runs") or []):
        k = int(row["K"])
        ckpt = row.get("ckpt", "")
        if task == "cls":
            c = row.get("cls_trained") or {}
            if c.get("bpfp_with_match_side") is None or c.get("acc") is None:
                continue
            pts.append((k, float(c["bpfp_with_match_side"]),
                          float(c["acc"]), ckpt))
        elif task == "depth":
            bm = row.get("depth_merge_rate") or {}
            if (row.get("depth_merge_rmse") is None
                    or bm.get("bpfp_with_match_side") is None):
                continue
            pts.append((k, float(bm["bpfp_with_match_side"]),
                          float(row["depth_merge_rmse"]), ckpt))
        elif task == "seg":
            sm = row.get("seg_merge") or {}
            if sm.get("bpfp_with_match_side") is None or sm.get("miou") is None:
                continue
            pts.append((k, float(sm["bpfp_with_match_side"]),
                          float(sm["miou"]), ckpt))
    return pareto_front(pts, minimize)


def _ylim(ys, cap=None):
    import math
    y_lo, y_hi = min(ys), max(ys)
    span = y_hi - y_lo
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


def _y_locator(lo, hi):
    span = hi - lo
    if span <= 0.04:
        step = 0.01
    elif span <= 0.10:
        step = 0.02
    elif span <= 0.25:
        step = 0.04
    else:
        step = 0.05
    return MultipleLocator(step)


def _xy(pts):
    if not pts:
        return [], []
    return [p[1] for p in pts], [p[2] for p in pts]


def plot_rd(orfc, merge, out_path: Path, title, ylabel, y_locator=None,
            y_cap=None, hline=None, hline_label=None, legend_loc="lower right",
            tome=None, trained=None, merge50=None, tome50=None,
            merge_label="greedy 0.25", tome_label="ToMe 0.25"):
    fig, ax = plt.subplots(figsize=(5.0, 3.7), dpi=160)
    ox, oy = _xy(orfc)
    rx, ry = _xy(merge)
    ys = oy + ry
    xs = ox + rx
    tx, ty = _xy(tome or [])
    if tome:
        ys, xs = ys + ty, xs + tx
    gx, gy = _xy(merge50 or [])
    if merge50:
        ys, xs = ys + gy, xs + gx
    hx, hy = _xy(tome50 or [])
    if tome50:
        ys, xs = ys + hy, xs + hx
    nx, ny = _xy(trained or [])
    if trained:
        ys, xs = ys + ny, xs + nx
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
    if merge:
        ax.plot(
            rx, ry, color=MERGE_COLOR, marker="s", markersize=7,
            markeredgecolor="white", markeredgewidth=0.6, linewidth=1.4,
            zorder=3, label=merge_label,
        )
    if tome:
        ax.plot(
            tx, ty, color=TOME_COLOR, marker="^", markersize=7,
            markeredgecolor="white", markeredgewidth=0.6, linewidth=1.4,
            zorder=3, label=tome_label,
        )
    if merge50:
        ax.plot(
            gx, gy, color=MERGE_COLOR, marker="s", markersize=7,
            markerfacecolor="white", markeredgecolor=MERGE_COLOR,
            markeredgewidth=1.0, linewidth=1.4, linestyle="--",
            zorder=3, label="greedy 0.50",
        )
    if tome50:
        ax.plot(
            hx, hy, color=TOME_COLOR, marker="^", markersize=7,
            markerfacecolor="white", markeredgecolor=TOME_COLOR,
            markeredgewidth=1.0, linewidth=1.4, linestyle="--",
            zorder=3, label="ToMe 0.50",
        )
    if trained:
        ax.plot(
            nx, ny, color=TRAINED_COLOR, marker="D", markersize=6.5,
            markeredgecolor="white", markeredgewidth=0.6, linewidth=1.4,
            zorder=3, label="ToMe + train R/PQ",
        )
    ax.set_xlabel("rANS BPFP")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    y_lo, y_hi = _ylim(ys, cap=y_cap)
    ax.set_ylim(y_lo, y_hi)
    xmin, xmax = min(xs), max(xs)
    pad = 0.08 * (xmax - xmin) if xmax > xmin else 0.01
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    loc = y_locator if y_locator is not None else _y_locator(y_lo, y_hi)
    ax.yaxis.set_major_locator(loc)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(frameon=False, loc=legend_loc, fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _task_style(task):
    if task == "depth":
        return {
            "ylabel": "depth RMSE",
            "title_task": "depth",
            "y_cap": None,
            "legend_loc": "upper right",
            "metric": "RMSE",
        }
    if task == "seg":
        return {
            "ylabel": "seg mIoU",
            "title_task": "segmentation",
            "y_cap": 1.0,
            "legend_loc": "lower right",
            "metric": "mIoU",
        }
    return {
        "ylabel": "cls Acc",
        "title_task": "classification",
        "y_cap": 1.0,
        "legend_loc": "lower right",
        "metric": "Acc",
    }


def write_csv(result_dir: Path, out_csv: Path, trained_dir: Path | None = None):
    by_key = {}
    fps = list(result_dir.glob("frozen_orfc_merge_*_shareR_*.json"))
    fps += list(result_dir.glob("frozen_tome_merge_*_shareR_*.json"))
    for fp in sorted(fps):
        d = json.loads(fp.read_text())
        layer = d["config"]["layer"]
        for row in d.get("runs") or []:
            key = (layer, row["ckpt"])
            rec = by_key.setdefault(key, {
                "layer": layer, "K": row["K"], "ckpt": row["ckpt"],
            })
            frac = _merge_frac(d, fp)
            tag = "" if _frac_close(frac, FRAC_025) else "_f05"
            is_tome = fp.name.startswith("frozen_tome_merge_")
            if "cls_orfc" in row:
                rec.update({
                    "cls_orfc_bpfp": row["cls_orfc"]["bpfp"],
                    "cls_orfc_acc": row["cls_orfc"]["acc"],
                })
            if "cls_merge" in row:
                prefix = "cls_tome" if is_tome else "cls_merge"
                rec.update({
                    f"{prefix}{tag}_bpfp": row["cls_merge"]["bpfp_with_match_side"],
                    f"{prefix}{tag}_acc": row["cls_merge"]["acc"],
                })
            if "depth_orfc_rmse" in row:
                rec.update({
                    "depth_orfc_bpfp": row["depth_orfc_rate"]["bpfp"],
                    "depth_orfc_rmse": row["depth_orfc_rmse"],
                })
            if row.get("depth_merge_rmse") is not None:
                bpfp = (row.get("depth_merge_rate") or {}).get(
                    "bpfp_with_match_side")
                prefix = "depth_tome" if is_tome else "depth_merge"
                rec.update({
                    f"{prefix}{tag}_bpfp": bpfp,
                    f"{prefix}{tag}_rmse": row["depth_merge_rmse"],
                })
            if "seg_orfc" in row:
                rec.update({
                    "seg_orfc_bpfp": row["seg_orfc"].get("bpfp"),
                    "seg_orfc_miou": row["seg_orfc"]["miou"],
                })
            if "seg_merge" in row:
                sm = row["seg_merge"]
                prefix = "seg_tome" if is_tome else "seg_merge"
                rec.update({
                    f"{prefix}{tag}_bpfp": sm.get("bpfp_with_match_side"),
                    f"{prefix}{tag}_miou": sm["miou"],
                })
    trained_by_lk = {}
    if trained_dir is not None:
        for fp in sorted(trained_dir.glob("eval_tasks_*_tome.json")):
            d = json.loads(fp.read_text())
            layer = d.get("config", {}).get("layer") or fp.stem.split("_")[2]
            for row in _pick_trained_runs(d.get("runs") or []):
                trained_by_lk[(layer, int(row["K"]))] = row
        for rec in by_key.values():
            t = trained_by_lk.get((rec["layer"], int(rec["K"])))
            if not t:
                continue
            c = t.get("cls_trained") or {}
            if c:
                rec.update({
                    "cls_trained_bpfp": c.get("bpfp_with_match_side"),
                    "cls_trained_acc": c.get("acc"),
                })
            if t.get("depth_merge_rmse") is not None:
                rec.update({
                    "depth_trained_bpfp": (
                        t.get("depth_merge_rate") or {}).get("bpfp_with_match_side"),
                    "depth_trained_rmse": t["depth_merge_rmse"],
                })
            sm = t.get("seg_merge") or {}
            if sm:
                rec.update({
                    "seg_trained_bpfp": sm.get("bpfp_with_match_side"),
                    "seg_trained_miou": sm.get("miou"),
                })
    rows = list(by_key.values())
    if not rows:
        return None
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return out_csv


def plot_paper_set(result_dir: Path, out_dir: Path, layers, include_tome=False):
    """ORFC + greedy merge (+ ToMe) + original-feature hline (depth / seg)."""
    for task in ("depth", "seg"):
        st = _task_style(task)
        for layer in layers:
            frac = PAPER_FRAC[layer]
            orfc, _, nyu_anchor = load_merge_points(result_dir, layer, task)
            greedy = load_greedy_frac_points(result_dir, layer, task, frac)
            tome = (load_tome_points(result_dir, layer, task, frac)
                    if include_tome else None)
            if not orfc or not greedy:
                print(f"skip {layer} {task}: missing ORFC or greedy f={frac:g}")
                continue
            if include_tome and not tome:
                print(f"skip {layer} {task}: missing ToMe f={frac:g}")
                continue
            if task == "depth":
                anchor = nyu_anchor
            else:
                anchor = load_voc_anchor(layer)
            stem = (f"{layer}_{task}_orfc_greedy_tome" if include_tome
                    else f"{layer}_{task}_orfc_greedy")
            out = out_dir / f"{stem}.png"
            merge_label = f"greedy merge (f={frac:g})"
            tome_label = f"ToMe (f={frac:g})"
            plot_rd(
                orfc, greedy, out,
                title=f"{layer}  {st['title_task']}",
                ylabel=st["ylabel"],
                y_cap=st["y_cap"],
                hline=anchor,
                hline_label="original",
                legend_loc=st["legend_loc"],
                tome=tome or None,
                merge_label=merge_label,
                tome_label=tome_label,
            )
            print(f"{layer} {task}  ORFC n={len(orfc)}  K={[p[0] for p in orfc]}")
            print(f"{layer} {task}  {merge_label} n={len(greedy)}  "
                  f"K={[p[0] for p in greedy]}")
            if tome:
                print(f"{layer} {task}  {tome_label} n={len(tome)}  "
                      f"K={[p[0] for p in tome]}")
            if anchor is not None:
                print(f"  original {st['metric']}={anchor:.4f}")
            series = [("ORFC", orfc), ("greedy", greedy)]
            if tome:
                series.append(("ToMe", tome))
            for name, pts in series:
                for k, x, y, *_rest in pts:
                    print(f"  {name:6s} K{k:<3d}  BPFP={x:.4f}  "
                          f"{st['metric']}={y:.4f}")
            print(out)


def plot_layer(layer, task, orfc, merge, out_dir: Path, anchor=None,
              tome=None, trained=None, merge50=None, tome50=None):
    st = _task_style(task)
    out = out_dir / f"{layer}_{task}_orfc_merge.png"
    plot_rd(
        orfc, merge, out,
        title=f"{layer}  {st['title_task']}",
        ylabel=st["ylabel"],
        y_cap=st["y_cap"],
        hline=anchor if task == "depth" else None,
        hline_label="unquantized",
        legend_loc=st["legend_loc"],
        tome=tome or None,
        trained=trained or None,
        merge50=merge50 or None,
        tome50=tome50 or None,
    )
    print(f"{layer} {task}  ORFC n={len(orfc)}  K={[p[0] for p in orfc]}")
    print(f"{layer} {task}  greedy0.25 n={len(merge)}  K={[p[0] for p in merge]}")
    if tome:
        print(f"{layer} {task}  ToMe0.25 n={len(tome)}  K={[p[0] for p in tome]}")
    if merge50:
        print(f"{layer} {task}  greedy0.50 n={len(merge50)}  K={[p[0] for p in merge50]}")
    if tome50:
        print(f"{layer} {task}  ToMe0.50 n={len(tome50)}  K={[p[0] for p in tome50]}")
    if trained:
        print(f"{layer} {task}  train n={len(trained)}  K={[p[0] for p in trained]}")
    series = [("ORFC", orfc), ("g0.25", merge)]
    if tome:
        series.append(("t0.25", tome))
    if merge50:
        series.append(("g0.50", merge50))
    if tome50:
        series.append(("t0.50", tome50))
    if trained:
        series.append(("train", trained))
    for name, pts in series:
        for k, x, y, *_rest in pts:
            print(f"  {name:6s} K{k:<3d}  BPFP={x:.4f}  {st['metric']}={y:.4f}")
    print(out)
    return out


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--result_dir",
        default=str(here / "results" / "similarity_merge" / "dinov2_vitl14"),
    )
    p.add_argument(
        "--out_dir",
        default=str(here / "results" / "similarity_merge" / "dinov2_vitl14" / "plots"),
    )
    p.add_argument(
        "--trained_dir",
        default=str(here / "results" / "similarity_merge_pq" / "dinov2_vitl14"),
    )
    p.add_argument("--layers", nargs="+", default=list(LAYERS))
    p.add_argument("--tasks", nargs="+", default=list(TASKS))
    p.add_argument(
        "--style", choices=("full", "paper", "paper_tome"), default="full",
        help="full: all series. paper: greedy+ORFC+original. "
             "paper_tome: also ToMe. blk05/10 f=0.25, blk15/20 f=0.5, "
             "depth+seg only.",
    )
    args = p.parse_args()
    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir)
    trained_dir = Path(args.trained_dir)

    if args.style in ("paper", "paper_tome"):
        include_tome = args.style == "paper_tome"
        if args.out_dir == p.get_default("out_dir"):
            out_dir = out_dir / ("greedy_tome" if include_tome else "greedy_orfc")
        plot_paper_set(result_dir, out_dir, args.layers,
                       include_tome=include_tome)
        return

    csv_path = write_csv(result_dir, out_dir / "orfc_merge_rd.csv",
                         trained_dir=trained_dir)
    if csv_path:
        print(csv_path)

    for task in args.tasks:
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}; expected {TASKS}")
        for layer in args.layers:
            orfc, merge, anchor = load_merge_points(result_dir, layer, task)
            tome = load_tome_points(result_dir, layer, task, frac=FRAC_025)
            merge50 = load_greedy_frac_points(result_dir, layer, task, FRAC_050)
            tome50 = load_tome_points(result_dir, layer, task, frac=FRAC_050)
            trained = load_trained_points(trained_dir, layer, task)
            if not orfc or not merge:
                print(f"skip {layer} {task}: no frozen merge json")
                continue
            plot_layer(layer, task, orfc, merge, out_dir, anchor=anchor,
                       tome=tome or None, trained=trained or None,
                       merge50=merge50 or None, tome50=tome50 or None)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""ShareR RD: frozen ORFC (blue) vs base+K2 residual (red).

One PNG per (layer, task).  Classification uses Acc (higher better);
depth uses NYU test80 RMSE (lower better) and draws the unquantized
anchor; segmentation uses VOC2012-100 mIoU with the same unquantized
hline; reconstruction uses RAEv2 ImageNet-500 PSNR (higher better)
with the bypass (unquantized) hline.  ORFC keeps Pareto points;
residual keeps the best run at each base K (so a later K that does
not raise mIoU / PSNR still appears).
The collapsed blk20 K4 (lr=3e-4 ep=300) is still dropped; the CSV
K4 (lr=5e-4) stays.

Style matches orfcv1/phase1/v34/plot_block_codec_rd.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

ORFC_COLOR = "#1f77b4"
RES_COLOR = "#d62728"
ANCHOR_COLOR = "0.45"

LAYERS = ("blk05", "blk10", "blk15", "blk20")
TASKS = ("cls", "depth", "seg", "recon")
D_FEAT = 1024.0

# DINOv3 RAEv2 residual sweep: K4/8/16 from λ=0.5 bases, K64/256 from λ=1.0.
RECON_SPECS = ((4, 0.5), (8, 0.5), (16, 0.5), (64, 1.0), (256, 1.0))
_CKPT_RE = re.compile(
    r"^(blk\d+)_K(\d+)_emb(\d+)_bt\d+_ws_lmbda([0-9.]+)_"
)

# Collapsed training run; keep the lr=5e-4 CoFAI/CSV checkpoint.
SKIP_STEM_SUBSTR = (
    "K4_emb32_bt1024_ws_tau0.5_lr0.0003_ep300",
    "smoke",
)

# VOC2012-100 mIoU of unquantized DINOv2-L replay (remaining ViT + linear
# head). Same at every layer; from eval_similarity_merge_seg.py.
_VOC_ANCHOR_JSON = (
    Path(__file__).resolve().parents[1]
    / "orfcv1" / "results" / "similarity_merge" / "dinov2_vitl14"
    / "voc100_blk05_blk10_blk15_blk20_f0.5-0.25.json"
)
VOC_UNQUANT_MIOU = 0.8165787


def voc_unquant_miou():
    fp = _VOC_ANCHOR_JSON
    if not fp.is_file():
        return VOC_UNQUANT_MIOU
    layers = json.loads(fp.read_text()).get("layers") or {}
    vals = [float(row["anchor_miou"]) for row in layers.values()
            if row.get("anchor_miou") is not None]
    return float(sum(vals) / len(vals)) if vals else VOC_UNQUANT_MIOU


def _base_k(ckpt_path: str) -> int:
    name = os.path.basename(ckpt_path)
    part = name.split("_")[1]  # K4 / K8 / ...
    return int(part[1:])


def _is_pure_kd(cfg: dict) -> bool:
    return float(cfg.get("ce_weight", 1.0)) == 0.0 and float(
        cfg.get("lmbda_kd", 0.0)) > 0.0


def _is_smoke(cfg: dict, stem: str) -> bool:
    if "smoke" in stem:
        return True
    if int(cfg.get("epochs", 0)) < 20:
        return True
    if 0 < int(cfg.get("max_train_images", 5000)) < 1000:
        return True
    return False


def pareto_front(points, minimize=False):
    """Keep undominated (rate, quality) points; quality monotone in rate."""
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


def best_per_k(points, minimize=False):
    """One point per base K, keeping the better quality if duplicates exist."""
    by_k = {}
    for p in points:
        k = p[0]
        if k not in by_k:
            by_k[k] = p
            continue
        better = p[2] < by_k[k][2] if minimize else p[2] > by_k[k][2]
        if better:
            by_k[k] = p
    return sorted(by_k.values(), key=lambda t: t[1])


def load_shareR_points(result_dir: Path, layer: str, task: str):
    orfc, residual = [], []
    anchor = None
    minimize = (task == "depth")
    for fp in sorted(result_dir.glob("*shareR*.json")):
        stem = fp.stem
        if any(s in stem for s in SKIP_STEM_SUBSTR):
            continue
        d = json.loads(fp.read_text())
        cfg = d["config"]
        if cfg.get("layer") != layer:
            continue
        if cfg.get("task", "cls") != task:
            continue
        if not cfg.get("share_base_transform"):
            continue
        if not _is_pure_kd(cfg):
            continue
        if int(cfg.get("K", 0)) != 2:
            continue
        if _is_smoke(cfg, stem):
            continue
        k = _base_k(cfg["base_ckpt"])
        bpfp_base = float(d["rate_base"]["rans_bpt"]) / 1024.0
        bpfp_full = float(d["rate_total_bpfp"])
        if task == "depth":
            if "base_rmse" not in d or "full_rmse" not in d:
                continue
            orfc.append((k, bpfp_base, float(d["base_rmse"])))
            residual.append((k, bpfp_full, float(d["full_rmse"])))
            if d.get("anchor_rmse") is not None:
                anchor = float(d["anchor_rmse"])
        elif task == "seg":
            if "base_miou" not in d or "full_miou" not in d:
                continue
            voc_base = d.get("voc_rate_base") or {}
            if voc_base.get("rans_bpt") is not None and d.get(
                    "voc_rate_total_rans_bpt") is not None:
                bpfp_base = float(voc_base["rans_bpt"]) / 1024.0
                bpfp_full = float(d["voc_rate_total_rans_bpt"]) / 1024.0
            orfc.append((k, bpfp_base, float(d["base_miou"])))
            residual.append((k, bpfp_full, float(d["full_miou"])))
            if d.get("anchor_miou") is not None:
                anchor = float(d["anchor_miou"])
        else:
            if "base_acc" not in d or "full_acc" not in d:
                continue
            orfc.append((k, bpfp_base, float(d["base_acc"])))
            residual.append((k, bpfp_full, float(d["full_acc"])))
    orfc = pareto_front(orfc, minimize)
    residual = best_per_k(residual, minimize)
    if task != "seg":
        residual = pareto_front(residual, minimize)
    if task == "seg" and anchor is None:
        anchor = voc_unquant_miou()
    return orfc, residual, anchor


def _parse_recon_tag(tag: str):
    m = _CKPT_RE.match(tag)
    if not m:
        return None
    return m.group(1), int(m.group(2)), float(m.group(4))


def _recon_spec_ok(k: int, lmbda: float) -> bool:
    return any(kk == k and abs(lmbda - ll) < 1e-6 for kk, ll in RECON_SPECS)


def load_recon_points(result_dir: Path, layer: str):
    """RAEv2 ImageNet-500 recon jsons: K1 (blue) vs shareR residual K2 (red)."""
    orfc, residual = [], []
    anchor = None
    for fp in sorted(result_dir.glob("recon_*.json")):
        if fp.stem.startswith("recon_bypass"):
            continue
        d = json.loads(fp.read_text())
        if d.get("layer") != layer or d.get("mode") != "orfc":
            continue
        if int(d.get("n_samples") or 0) < 400:
            continue
        tag = d.get("tag") or fp.stem.replace("recon_", "", 1)
        parsed = _parse_recon_tag(tag)
        if parsed is None:
            continue
        lyr, k, lmbda = parsed
        if lyr != layer or not _recon_spec_ok(k, lmbda):
            continue
        psnr = (d.get("metrics") or {}).get("psnr")
        if psnr is None:
            continue
        rate = d.get("rate") or {}
        bpt = float(rate.get("bpfp") or 0.0)
        if bpt <= 0:
            continue
        bpfp = bpt / D_FEAT
        is_res = bool(rate.get("residual")) or "resK2" in tag
        if is_res:
            residual.append((k, bpfp, float(psnr)))
        else:
            orfc.append((k, bpfp, float(psnr)))
        bp = (d.get("metrics_bypass") or {}).get("psnr")
        if bp is not None:
            anchor = float(bp)
    orfc = pareto_front(orfc, minimize=False)
    residual = best_per_k(residual, minimize=False)
    return orfc, residual, anchor


def _ylim(ys, cap=None, psnr=False):
    y_lo, y_hi = min(ys), max(ys)
    span = max(y_hi - y_lo, 1e-9)
    if psnr:
        pad = max(0.3, 0.08 * span)
        lo = math.floor((y_lo - pad) * 2 - 1e-9) / 2
        hi = math.ceil((y_hi + pad) * 2 + 1e-9) / 2
        if cap is not None:
            hi = min(hi, cap)
        if hi - lo < 1.0:
            mid = 0.5 * (y_lo + y_hi)
            lo, hi = mid - 0.5, mid + 0.5
            if cap is not None:
                hi = min(hi, cap)
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


def _y_locator(lo, hi, psnr=False):
    span = hi - lo
    if psnr:
        return MultipleLocator(0.5 if span <= 4 else 1.0)
    if span <= 0.04:
        step = 0.01
    elif span <= 0.10:
        step = 0.02
    elif span <= 0.25:
        step = 0.04
    else:
        step = 0.05
    return MultipleLocator(step)


def plot_rd(orfc, residual, out_path: Path, title, ylabel, y_locator=None,
            y_cap=None, hline=None, hline_label=None, legend_loc="lower right",
            psnr=False):
    fig, ax = plt.subplots(figsize=(4.8, 3.6), dpi=160)
    ox, oy = [p[1] for p in orfc], [p[2] for p in orfc]
    rx, ry = [p[1] for p in residual], [p[2] for p in residual]
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
        rx, ry, color=RES_COLOR, marker="s", markersize=7,
        markeredgecolor="white", markeredgewidth=0.6, linewidth=1.4,
        zorder=3, label="residual K2",
    )

    ax.set_xlabel("rANS BPFP")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    y_lo, y_hi = _ylim(ys, cap=y_cap, psnr=psnr)
    ax.set_ylim(y_lo, y_hi)
    xmin, xmax = min(ox + rx), max(ox + rx)
    pad = 0.08 * (xmax - xmin) if xmax > xmin else 0.01
    ax.set_xlim(xmin - pad, xmax + pad)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    loc = y_locator if y_locator is not None else _y_locator(y_lo, y_hi, psnr=psnr)
    ax.yaxis.set_major_locator(loc)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f" if psnr else "%.2f"))
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend(frameon=False, loc=legend_loc)
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
            "hline": True,
            "psnr": False,
        }
    if task == "seg":
        return {
            "ylabel": "seg mIoU",
            "title_task": "segmentation",
            "y_cap": 1.0,
            "legend_loc": "lower right",
            "metric": "mIoU",
            "hline": True,
            "psnr": False,
        }
    if task == "recon":
        return {
            "ylabel": "recon PSNR",
            "title_task": "reconstruction",
            "y_cap": None,
            "legend_loc": "lower right",
            "metric": "PSNR",
            "hline": True,
            "psnr": True,
        }
    return {
        "ylabel": "cls Acc",
        "title_task": "classification",
        "y_cap": 1.0,
        "legend_loc": "lower right",
        "metric": "Acc",
        "hline": False,
        "psnr": False,
    }


def plot_layer(layer, task, orfc, residual, out_dir: Path, anchor=None):
    st = _task_style(task)
    out = out_dir / f"{layer}_{task}_shareR_k2.png"
    y_loc = MultipleLocator(0.02) if (layer == "blk20" and task == "cls") else None
    plot_rd(
        orfc, residual, out,
        title=f"{layer}  {st['title_task']}",
        ylabel=st["ylabel"],
        y_locator=y_loc,
        y_cap=st["y_cap"],
        hline=anchor if st["hline"] else None,
        hline_label="unquantized",
        legend_loc=st["legend_loc"],
        psnr=st["psnr"],
    )
    print(f"{layer} {task}  ORFC n={len(orfc)}  K={[p[0] for p in orfc]}")
    print(f"{layer} {task}  res  n={len(residual)}  K={[p[0] for p in residual]}")
    if anchor is not None:
        print(f"  unquantized {st['metric']}={anchor:.4f}")
    for name, pts in (("ORFC", orfc), ("res", residual)):
        for k, x, y in pts:
            print(f"  {name:6s} K{k:<3d}  BPFP={x:.4f}  {st['metric']}={y:.4f}")
    print(out)
    return out


def main():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--result_dir",
        default=str(here / "results" / "dinov2_vitl14"),
    )
    p.add_argument(
        "--out_dir",
        default=str(here / "results" / "dinov2_vitl14" / "plots"),
    )
    p.add_argument(
        "--recon_dir",
        default=str(
            here.parent / "orfc" / "eval" / "results" / "dinov3_vitl16_raev2"
        ),
        help="RAEv2 recon json dir (task=recon)",
    )
    p.add_argument(
        "--recon_out_dir",
        default=str(here / "results" / "dinov3_vitl16_raev2" / "plots"),
        help="Output dir for reconstruction RD plots",
    )
    p.add_argument(
        "--layers", nargs="+", default=list(LAYERS),
        help="Which layers to plot (default: blk05 blk10 blk15 blk20)",
    )
    p.add_argument(
        "--tasks", nargs="+", default=list(TASKS),
        help="Which tasks to plot (default: cls depth seg recon)",
    )
    args = p.parse_args()
    result_dir = Path(args.result_dir)
    out_dir = Path(args.out_dir)
    recon_dir = Path(args.recon_dir)
    recon_out = Path(args.recon_out_dir)

    for task in args.tasks:
        if task not in TASKS:
            raise ValueError(f"unknown task {task!r}; expected {TASKS}")
        for layer in args.layers:
            if task == "recon":
                orfc, residual, anchor = load_recon_points(recon_dir, layer)
                if not orfc or not residual:
                    print(f"skip {layer} recon: no RAEv2 recon json in {recon_dir}")
                    continue
                plot_layer(layer, task, orfc, residual, recon_out, anchor=anchor)
                continue
            orfc, residual, anchor = load_shareR_points(result_dir, layer, task)
            if not orfc or not residual:
                print(f"skip {layer} {task}: no pure-KD shareR K2 json")
                continue
            plot_layer(layer, task, orfc, residual, out_dir, anchor=anchor)


if __name__ == "__main__":
    main()

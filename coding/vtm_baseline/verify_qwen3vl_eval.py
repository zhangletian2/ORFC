#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen3-VL VTM 验证评测：加载模型一次，评测 baseline / anchor / 各 QP 回放精度。
同时计算 BPFP，输出汇总表。

由 verify_vtm_qwen3vl.sh 调用。
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

TOOLS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "tools")
sys.path.insert(0, TOOLS_DIR)
from qwen3vl_feat_pipeline import (
    load_model,
    load_mmbench,
    build_prompt,
    generate_answer,
    extract_mcq_answer,
    replay_hooks,
)


def read_bpfp(stats_root, qp, layer_name, stems):
    csv_path = os.path.join(stats_root, str(qp), "_stats.csv")
    if not os.path.exists(csv_path):
        return {}
    bpfps = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row.get("layer") == layer_name and row["filename"] in stems:
                bpfps[row["filename"]] = float(row["bpfp"])
    return bpfps


def eval_baseline(model, processor, ds, max_new_tokens, no_think):
    """直接推理（无 hook），作为精度基线。"""
    results = []
    correct = total = 0
    for sample in tqdm(ds, desc="baseline"):
        messages = build_prompt(sample)
        text, _ = generate_answer(model, processor, messages,
                                  max_new_tokens=max_new_tokens,
                                  no_think=no_think)
        pred = extract_mcq_answer(text)
        gt = sample["answer"]
        ok = pred == gt
        correct += int(ok)
        total += 1
        results.append(dict(index=str(sample["index"]), pred=pred, gt=gt,
                            correct=ok, raw=text[:300]))
    acc = correct / max(total, 1) * 100
    return dict(accuracy=acc, correct=correct, total=total, results=results)


def eval_replay(model, processor, vision_model, ds, feat_dir,
                layer, dtype, device, max_new_tokens, no_think, label):
    """加载保存的 .npy 特征并通过 hook 回放推理。"""
    results = []
    correct = total = 0
    for sample in tqdm(ds, desc=label):
        sid = str(sample["index"])
        feat_path = os.path.join(feat_dir, f"{sid}.npy")
        if not os.path.exists(feat_path):
            continue
        saved_blk = torch.from_numpy(np.load(feat_path))
        messages = build_prompt(sample)
        with replay_hooks(vision_model, saved_blk, layer, device, dtype):
            text, _ = generate_answer(model, processor, messages,
                                      max_new_tokens=max_new_tokens,
                                      no_think=no_think)
        pred = extract_mcq_answer(text)
        gt = sample["answer"]
        ok = pred == gt
        correct += int(ok)
        total += 1
        results.append(dict(index=sid, pred=pred, gt=gt,
                            correct=ok, raw=text[:300]))
    acc = correct / max(total, 1) * 100
    return dict(accuracy=acc, correct=correct, total=total, results=results)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--orig_feat_dir", required=True,
                    help="原始 .npy 特征目录")
    ap.add_argument("--decoded_root", required=True,
                    help="VTM 解码目录根，结构: {root}/{qp}/{layer}/*.npy")
    ap.add_argument("--stats_root", required=True,
                    help="VTM stats CSV 根目录")
    ap.add_argument("--qps", nargs="+", type=int, required=True)
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--output", required=True)
    ap.add_argument("--split", default="validation")
    ap.add_argument("--data_subdir", default="en")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--no_think", action="store_true")
    args = ap.parse_args()

    layer_name = f"blk{args.layer:02d}"

    stems = sorted(f.replace(".npy", "")
                   for f in os.listdir(args.orig_feat_dir)
                   if f.endswith(".npy"))
    print(f"验证样本 ({len(stems)}): {stems}")

    # --- 构造只含验证样本的临时 list ---
    tmp_list = os.path.join(os.path.dirname(args.output), "_verify_list.txt")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(tmp_list, "w") as f:
        for s in stems:
            f.write(f"{s}\tX\t4\tX\n")

    # --- 加载模型 ---
    print("\n加载 Qwen3-VL 模型...")
    t_load = time.time()
    model, processor = load_model(args.model_path, args.device, args.dtype)
    vision_model = model.model.visual
    dtype = next(vision_model.parameters()).dtype
    device = next(vision_model.parameters()).device
    print(f"  模型加载完成 ({time.time() - t_load:.1f}s)")

    ds = load_mmbench(args.data_dir, args.split, None, args.data_subdir,
                      None, tmp_list)
    print(f"  加载 {len(ds)} 条数据")

    # ── 1. Baseline（无 hook） ──
    print("\n── Baseline（直接推理）──")
    t0 = time.time()
    baseline = eval_baseline(model, processor, ds,
                             args.max_new_tokens, args.no_think)
    t_base = time.time() - t0
    print(f"  Acc={baseline['accuracy']:.1f}%  "
          f"({baseline['correct']}/{baseline['total']})  {t_base:.1f}s")

    # ── 2. Anchor（回放原始特征） ──
    print("\n── Anchor（回放原始 .npy）──")
    t0 = time.time()
    anchor = eval_replay(model, processor, vision_model, ds,
                         args.orig_feat_dir, args.layer, dtype, device,
                         args.max_new_tokens, args.no_think, "anchor")
    t_anc = time.time() - t0
    print(f"  Acc={anchor['accuracy']:.1f}%  "
          f"({anchor['correct']}/{anchor['total']})  {t_anc:.1f}s")

    # ── 3. 各 QP 回放 ──
    qp_data = {}
    for qp in args.qps:
        print(f"\n── QP={qp} ──")
        decoded_npy_dir = os.path.join(args.decoded_root, str(qp), layer_name)
        if not os.path.exists(decoded_npy_dir):
            print(f"  [SKIP] {decoded_npy_dir} not found")
            continue

        bpfps = read_bpfp(args.stats_root, qp, layer_name, set(stems))

        t0 = time.time()
        qp_eval = eval_replay(model, processor, vision_model, ds,
                               decoded_npy_dir, args.layer, dtype, device,
                               args.max_new_tokens, args.no_think,
                               f"QP{qp}")
        t_qp = time.time() - t0

        avg_bpfp = float(np.mean(list(bpfps.values()))) if bpfps else 0.0

        qp_eval.update(avg_bpfp=avg_bpfp, bpfps=bpfps,
                        elapsed_s=round(t_qp, 1))
        qp_data[qp] = qp_eval

        print(f"  Acc={qp_eval['accuracy']:.1f}%  "
              f"BPFP={avg_bpfp:.4f}  {t_qp:.1f}s")

    # ══════════════════════════════════════════════════
    #  Per-sample detail
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 90)
    print("  Per-sample Prediction Detail")
    print("=" * 90)

    qp_keys = sorted(qp_data.keys())
    header = f"{'Sample':>10}  {'GT':>3}  {'Base':>5}  {'Anchor':>7}"
    for qp in qp_keys:
        header += f"  {'QP' + str(qp):>6}"
    print(header)
    print("-" * 90)

    base_map = {r["index"]: r for r in baseline["results"]}
    anc_map = {r["index"]: r for r in anchor["results"]}
    qp_maps = {qp: {r["index"]: r for r in qp_data[qp]["results"]}
               for qp in qp_keys}

    for stem in stems:
        gt = base_map[stem]["gt"] if stem in base_map else "?"
        b_pred = base_map.get(stem, {}).get("pred", "?")
        a_pred = anc_map.get(stem, {}).get("pred", "?")
        b_ok = "✓" if b_pred == gt else "✗"
        a_ok = "✓" if a_pred == gt else "✗"
        row = f"{stem:>10}  {gt:>3}  {b_pred}({b_ok:1s})  {a_pred}  ({a_ok:1s})"
        for qp in qp_keys:
            p = qp_maps[qp].get(stem, {}).get("pred", "?")
            ok = "✓" if p == gt else "✗"
            row += f"  {p}({ok:1s}) "
        print(row)

    # ══════════════════════════════════════════════════
    #  Summary Table
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 64)
    print("  Qwen3-VL VTM Codec 验证汇总")
    print("=" * 64)
    print(f"{'Mode':>8}  {'BPFP':>8}  {'Acc%':>7}  {'ΔAcc':>7}  {'Correct':>8}")
    print("-" * 64)
    print(f"{'base':>8}  {'--':>8}  "
          f"{baseline['accuracy']:>6.1f}%  {'--':>7}  "
          f"{baseline['correct']}/{baseline['total']}")
    print(f"{'anchor':>8}  {'--':>8}  "
          f"{anchor['accuracy']:>6.1f}%  "
          f"{anchor['accuracy'] - baseline['accuracy']:>+6.1f}%  "
          f"{anchor['correct']}/{anchor['total']}")
    for qp in qp_keys:
        r = qp_data[qp]
        d_acc = r["accuracy"] - baseline["accuracy"]
        print(f"{'QP' + str(qp):>8}  {r['avg_bpfp']:>8.4f}  "
              f"{r['accuracy']:>6.1f}%  {d_acc:>+6.1f}%  "
              f"{r['correct']}/{r['total']}")
    print("=" * 64)

    # ── Save JSON ──
    output = dict(
        baseline=baseline, anchor=anchor,
        qp_results={str(k): v for k, v in qp_data.items()},
        verify_stems=stems,
    )
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  → 结果保存至 {args.output}")


if __name__ == "__main__":
    main()

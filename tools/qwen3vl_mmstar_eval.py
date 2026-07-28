#!/usr/bin/env python3
"""
Qwen3-VL-8B-Thinking MMStar 评测 (transformers 推理)

Prompt 对齐 VLMEvalKit 格式:
    Question: {question}
    Options:
    A. {option_a}
    B. {option_b}
    C. {option_c}
    D. {option_d}
    Answer with the option letter only.

参考: https://github.com/EvolvingLMMs-Lab/lmms-eval/issues/901
      https://github.com/open-compass/VLMEvalKit/blob/main/vlmeval/vlm/qwen3_vl/prompt.py
"""

import os
import sys
import json
import time
import re
import argparse
from io import BytesIO
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════
# Prompt 构造 (对齐 VLMEvalKit)
# ═══════════════════════════════════════════════════════════════

def parse_options_from_question(question_text):
    """解析 MMStar 合并格式: "question\\nOptions: A: ..., B: ..., C: ..., D: ..."."""
    if "Options:" not in question_text:
        return question_text.strip(), {}

    parts = question_text.split("Options:", 1)
    question = parts[0].strip()
    options_str = parts[1].strip()

    options = {}
    pattern = r'([A-D]):\s*(.+?)(?=,\s*[A-D]:|$)'
    for letter, value in re.findall(pattern, options_str):
        options[letter] = value.strip().rstrip(",").strip()

    return question, options


def build_vlmevalkit_prompt(question, options):
    """VLMEvalKit 风格 prompt（与官方 Qwen3-VL 评测对齐）."""
    parts = [f"Question: {question}"]
    if options:
        parts.append("Options:")
        for letter in sorted(options.keys()):
            parts.append(f"{letter}. {options[letter]}")
    parts.append("Answer with the option letter only.")
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════
# 答案抽取 (与 qwen3vl_feat_pipeline.py 对齐)
# ═══════════════════════════════════════════════════════════════

_ANSWER_PHRASES = [
    "the answer is", "answer is", "the correct answer is",
    "correct answer is", "the best answer is", "best answer is",
    "i choose", "i select", "i pick", "my answer is",
]

_FMT_PRIO = {
    "start": 10, "end": 9, "phrase": 7, "parentheses": 6,
    "period": 5, "colon": 4, "right_paren": 3, "space": 2, "fallback": 0,
}


def extract_answer(response, choices=None):
    """从模型输出中提取 MCQ 答案，自动剥离 <think>...</think>。"""
    if not response or not response.strip():
        return ""
    if "</think>" in response:
        response = response.split("</think>")[-1]

    choices = choices or ["A", "B", "C", "D"]
    text = response.strip()

    if text in choices:
        return text

    for ch in [",", ".", "!", "?", ";", ":", "'", '"']:
        text = text.strip(ch)
    text = " " + text + " "

    cands = []
    for c in choices:
        if f"({c})" in text:
            cands.append((c, text.rfind(f"({c})"), "parentheses"))
    for c in choices:
        if f"{c}." in text:
            cands.append((c, text.rfind(f"{c}."), "period"))
    for c in choices:
        if f"{c}:" in text:
            cands.append((c, text.rfind(f"{c}:"), "colon"))
    for c in choices:
        if f"{c})" in text:
            cands.append((c, text.rfind(f"{c})"), "right_paren"))
    for c in choices:
        if f"{c} " in text:
            cands.append((c, text.rfind(f"{c} "), "space"))

    text_lower = text.lower()
    for phrase in _ANSWER_PHRASES:
        idx = text_lower.find(phrase)
        if idx != -1:
            after = idx + len(phrase)
            for c in choices:
                pos = text.find(c, after)
                if pos != -1:
                    cands.append((c, pos, "phrase"))

    stripped = text.strip()
    for c in choices:
        if stripped.startswith(c) and (len(stripped) == 1 or not stripped[1].isalpha()):
            cands.append((c, 0, "start"))
    for c in choices:
        if stripped.endswith(c) and (len(stripped) == 1 or not stripped[-2].isalpha()):
            cands.append((c, len(text) - 1, "end"))

    if not cands:
        for c in choices:
            if c in text:
                cands.append((c, text.rfind(c), "fallback"))

    if not cands:
        return ""
    cands.sort(key=lambda x: (_FMT_PRIO.get(x[2], 0), x[1]), reverse=True)
    return cands[0][0]


# ═══════════════════════════════════════════════════════════════
# 模型加载 & 推理
# ═══════════════════════════════════════════════════════════════

def load_model(model_path, device="cuda", dtype="bf16"):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

    torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                   "fp32": torch.float32}[dtype]
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch_dtype, device_map=device,
    ).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor



@torch.no_grad()
def generate_answer(model, processor, messages, max_new_tokens=4096,
                    temperature=1.0, top_p=0.95, top_k=20,
                    repetition_penalty=1.0, do_sample=True):
    """单条推理，返回 (解码文本, 生成 token 数)。"""
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v
              for k, v in inputs.items()}

    gen_kwargs = dict(max_new_tokens=max_new_tokens)
    if do_sample:
        gen_kwargs.update(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
    else:
        gen_kwargs["do_sample"] = False

    output_ids = model.generate(**inputs, **gen_kwargs)
    input_len = inputs["input_ids"].shape[1]
    gen_ids = output_ids[0, input_len:]
    gen_token_count = len(gen_ids)
    text = processor.decode(gen_ids, skip_special_tokens=False)
    return text, gen_token_count


# ═══════════════════════════════════════════════════════════════
# 主流程
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser("Qwen3-VL MMStar Eval (transformers)")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--data_path", required=True,
                    help="MMStar parquet 文件路径")
    ap.add_argument("--output", required=True,
                    help="JSON 结果输出路径")
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=40960)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16",
                    choices=["bf16", "fp16", "fp32"])
    # 官方推理参数
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--repetition_penalty", type=float, default=1.0)
    ap.add_argument("--greedy", action="store_true",
                    help="使用 greedy 解码（覆盖 temperature/top_p/top_k）")
    args = ap.parse_args()

    do_sample = not args.greedy

    # ── 加载数据 ──
    df = pd.read_parquet(args.data_path)
    if args.max_samples:
        df = df.head(args.max_samples)
    print(f"[data] {len(df)} samples loaded from MMStar")

    # ── 加载模型 ──
    print(f"[model] Loading {args.model_path} ...")
    model, processor = load_model(args.model_path, args.device, args.dtype)
    print(f"[model] Loaded. dtype={args.dtype}, device={args.device}")

    # ── 打印配置 ──
    sample_str = (f"temperature={args.temperature}, top_p={args.top_p}, "
                  f"top_k={args.top_k}, rep_penalty={args.repetition_penalty}"
                  if do_sample else "greedy")
    print(f"[config] thinking=ON (always), sampling={sample_str}, "
          f"max_new_tokens={args.max_new_tokens}")

    # ── 逐样本推理 ──
    results = []
    correct = 0
    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    gen_token_counts = []
    t0 = time.time()

    for idx_row, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df),
                                            desc="mmstar_eval")):
        question, options = parse_options_from_question(row["question"])
        prompt_text = build_vlmevalkit_prompt(question, options)

        img = Image.open(BytesIO(row["image"])).convert("RGB")

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt_text},
            ],
        }]

        response, gen_tokens = generate_answer(
            model, processor, messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            do_sample=do_sample,
        )
        gen_token_counts.append(gen_tokens)

        pred = extract_answer(response)
        gt = row["answer"]
        hit = pred == gt
        correct += int(hit)

        cat = row.get("category", "unknown")
        cat_stats[cat]["correct"] += int(hit)
        cat_stats[cat]["total"] += 1

        results.append({
            "index": int(row["index"]),
            "pred": pred,
            "gt": gt,
            "correct": hit,
            "category": cat,
            "l2_category": row.get("l2_category", ""),
            "gen_tokens": gen_tokens,
            "response": response,
        })

        if (idx_row + 1) % 50 == 0:
            acc_so_far = correct / (idx_row + 1) * 100
            vram_gb = torch.cuda.max_memory_allocated() / 1024**3
            avg_tok = np.mean(gen_token_counts)
            print(f"  [{idx_row+1}/{len(df)}] acc={acc_so_far:.2f}%  "
                  f"avg_tokens={avg_tok:.0f}  peak_vram={vram_gb:.2f}GB")

    elapsed = time.time() - t0
    total = len(df)
    acc = correct / max(total, 1) * 100
    vram_gb = (torch.cuda.max_memory_allocated() / 1024**3
               if torch.cuda.is_available() else 0)

    # ── 分类精度 ──
    cat_acc = {}
    for cat in sorted(cat_stats):
        s = cat_stats[cat]
        cat_acc[cat] = round(s["correct"] / max(s["total"], 1) * 100, 2)

    # ── Token 生成长度分布 ──
    tok_arr = np.array(gen_token_counts)
    token_stats = {
        "min": int(tok_arr.min()),
        "max": int(tok_arr.max()),
        "mean": round(float(tok_arr.mean()), 1),
        "median": round(float(np.median(tok_arr)), 1),
        "p25": round(float(np.percentile(tok_arr, 25)), 1),
        "p75": round(float(np.percentile(tok_arr, 75)), 1),
        "p95": round(float(np.percentile(tok_arr, 95)), 1),
        "p99": round(float(np.percentile(tok_arr, 99)), 1),
        "total": int(tok_arr.sum()),
    }

    # ── 打印结果 ──
    print(f"\n{'=' * 65}")
    print(f"  MMStar Overall Accuracy: {acc:.2f}% ({correct}/{total})")
    print(f"  Time: {elapsed:.1f}s  ({elapsed/max(total,1):.2f}s/sample)")
    print(f"  Peak VRAM: {vram_gb:.2f}GB")
    print(f"  Thinking: ON (always)")
    print(f"  Sampling: {sample_str}")
    print(f"\n  Per-category accuracy:")
    for cat, a in cat_acc.items():
        n = cat_stats[cat]["total"]
        print(f"    {cat:30s}: {a:6.2f}%  ({cat_stats[cat]['correct']}/{n})")
    print(f"\n  Token generation stats:")
    print(f"    min={token_stats['min']}  p25={token_stats['p25']}  "
          f"median={token_stats['median']}  mean={token_stats['mean']}  "
          f"p75={token_stats['p75']}  p95={token_stats['p95']}  "
          f"max={token_stats['max']}")
    print(f"    total_tokens={token_stats['total']}")
    print(f"{'=' * 65}")

    # ── 保存 ──
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    payload = {
        "model": args.model_path,
        "accuracy": round(acc, 2),
        "total": total,
        "correct": correct,
        "elapsed_s": round(elapsed, 2),
        "speed_s_per_sample": round(elapsed / max(total, 1), 2),
        "peak_vram_gb": round(vram_gb, 2),
        "config": {
            "thinking": True,
            "do_sample": do_sample,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
        },
        "category_accuracy": cat_acc,
        "token_stats": token_stats,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

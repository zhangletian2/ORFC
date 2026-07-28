# -*- coding: utf-8 -*-
"""
Qwen3-VL-8B-Instruct  MMStar 分层抽取 / 分层回放 VQA Pipeline：

- baseline：直接跑 MMStar 评测作为精度基线。
- extract ：在 ViT 指定层（默认 block[8]）注册 forward hook，
            保存该层 hidden_states [N, 1152]。
- replay  ：加载保存的中间层特征（或 codec 重建特征），用 hook
            替换 ViT block 输出，后续的 DeepStack 和 block 均从
            重建特征自然重算 → merger → LLM generate。

设计意图：将 ViT 在 block[8] 处拆分为前后两半，前半 blocks 0-8
的输出经 codec 编解码后，送入后半 blocks 9-26 + DeepStack + merger
完成端到端推理。

Qwen3-VL-8B-Instruct 视觉塔参数：
  depth=27, hidden_size=1152, patch_size=16, spatial_merge_size=2
  deepstack_visual_indexes=[8, 16, 24]
  merger → out_hidden_size=4096

Prompt 对齐 https://github.com/QwenLM/Qwen3-VL:
  <image>
  Question: {question}
  Options:
  A. ... B. ... C. ... D. ...
  Please select the correct answer from the options above.

官方 Instruct 推理参数:
  seed=3407, temperature=0.7, top_p=0.8, top_k=20,
  repetition_penalty=1.0, presence_penalty=1.5,
  out_seq_length=32768

依赖：transformers>=4.57, torch>=2.1, Pillow, pandas, numpy
"""

import os
import json
import time
import re
import argparse
from io import BytesIO
from contextlib import contextmanager
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from transformers import LogitsProcessor, LogitsProcessorList
from PIL import Image
from tqdm import tqdm


# ═══════════════════════════════════════════════════════════════
# 自定义 LogitsProcessor (presence_penalty)
# ═══════════════════════════════════════════════════════════════

class PresencePenaltyLogitsProcessor(LogitsProcessor):
    """OpenAI 风格的 presence_penalty：对已出现 token 的 logit 减去固定值。"""

    def __init__(self, penalty: float):
        self.penalty = penalty

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.penalty == 0.0:
            return scores
        generated = input_ids[0].unique()
        scores[0, generated] -= self.penalty
        return scores


# ═══════════════════════════════════════════════════════════════
# 模型 / 数据加载
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


def load_mmstar(parquet_path, max_samples=None, sample_list=None):
    """加载 MMStar parquet 数据集，返回 DataFrame。"""
    df = pd.read_parquet(parquet_path)

    if sample_list:
        index_set = _load_sample_list(sample_list)
        df = df[df["index"].isin(index_set)].reset_index(drop=True)
        print(f"[sample_list] filtered {len(df)} samples from {sample_list}")
    elif max_samples:
        df = df.head(max_samples)

    return df


def _load_sample_list(path):
    indices = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            indices.add(int(line.split("\t")[0]))
    return indices


# ═══════════════════════════════════════════════════════════════
# MMStar Prompt 构造 / 答案解析
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


def build_prompt(question, options):
    """对齐 https://github.com/QwenLM/Qwen3-VL 官方评测 prompt."""
    parts = [f"Question: {question}"]
    if options:
        parts.append("Options:")
        for letter in sorted(options.keys()):
            parts.append(f"{letter}. {options[letter]}")
    parts.append("Please select the correct answer from the options above.")
    return "\n".join(parts)


def build_messages(row):
    """将 MMStar 行数据构造为 Qwen3-VL chat messages 格式。"""
    question, options = parse_options_from_question(row["question"])
    prompt_text = build_prompt(question, options)

    img = Image.open(BytesIO(row["image"])).convert("RGB")

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": prompt_text},
        ],
    }]
    return messages


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
    """从模型输出中提取 MCQ 答案。"""
    if not response or not response.strip():
        return ""

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
        if stripped.startswith(c) and (len(stripped) == 1
                                       or not stripped[1].isalpha()):
            cands.append((c, 0, "start"))
    for c in choices:
        if stripped.endswith(c) and (len(stripped) == 1
                                     or not stripped[-2].isalpha()):
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
# Hook 工具：抽取 / 回放
# ═══════════════════════════════════════════════════════════════

class VisionBlockCatcher:
    """在指定 ViT block 上注册 forward hook，抓取输出 hidden_states。

    适用于 Qwen3-VL 全系列（4B/8B/...），仅依赖 blocks[idx]。
    """

    def __init__(self, blocks, layer_indices):
        self.indices = sorted(set(int(i) for i in layer_indices))
        self.buf = {}
        self._handles = []
        for idx in self.indices:
            key = f"blk{idx:02d}"

            def _hook(_mod, _inp, out, _key=key):
                self.buf[_key] = out.detach().cpu().float()

            self._handles.append(blocks[idx].register_forward_hook(_hook))

    def pop(self):
        out = self.buf
        self.buf = {}
        return out

    def close(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


@contextmanager
def replay_hooks(vision_model, saved_blk, layer, device, dtype):
    """
    Context manager：用 hook 替换指定 ViT block 的输出，
    使后续 block 和 DeepStack 均基于替换后的特征继续前向。

    saved_blk : Tensor [N, D] — block[layer] 的 hidden_states（或 codec 重建）

    当切分点恰好是 deepstack 层（如 layer=8 对应
    deepstack_visual_indexes[0]）时，deepstack merger 会从替换后的
    hidden_states 自然重算，确保下游链路完整反映 codec 影响。
    """
    blk_tensor = saved_blk.to(device=device, dtype=dtype)

    def _blk_hook(_mod, _inp, _out):
        return blk_tensor

    handle = vision_model.blocks[layer].register_forward_hook(_blk_hook)
    try:
        yield
    finally:
        handle.remove()


# ═══════════════════════════════════════════════════════════════
# 单样本推理
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def generate_answer(model, processor, messages, max_new_tokens=32768,
                    temperature=0.7, top_p=0.8, top_k=20,
                    repetition_penalty=1.0, presence_penalty=1.5,
                    do_sample=True):
    """单条推理，返回 (解码文本, 生成 token 数, inputs)。"""
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

    if presence_penalty != 0.0:
        gen_kwargs["logits_processor"] = LogitsProcessorList([
            PresencePenaltyLogitsProcessor(presence_penalty),
        ])

    output_ids = model.generate(**inputs, **gen_kwargs)
    input_len = inputs["input_ids"].shape[1]
    gen_ids = output_ids[0, input_len:]
    gen_token_count = len(gen_ids)
    text = processor.decode(gen_ids, skip_special_tokens=True)
    return text, gen_token_count, inputs


# ═══════════════════════════════════════════════════════════════
# 子命令：baseline
# ═══════════════════════════════════════════════════════════════

def cmd_baseline(args):
    model, processor = load_model(args.model_path, args.device, args.dtype)
    df = load_mmstar(args.data_path, args.max_samples,
                     getattr(args, "sample_list", None))
    print(f"[data] {len(df)} samples loaded from MMStar")

    do_sample = not args.greedy
    correct = total = 0
    results = []
    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    gen_token_counts = []
    t0 = time.time()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="baseline"):
        messages = build_messages(row)
        text, gen_tokens, _ = generate_answer(
            model, processor, messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            presence_penalty=args.presence_penalty,
            do_sample=do_sample,
        )
        gen_token_counts.append(gen_tokens)

        pred = extract_answer(text)
        gt = row["answer"]
        hit = pred == gt
        correct += int(hit)
        total += 1

        cat = row.get("category", "unknown")
        cat_stats[cat]["correct"] += int(hit)
        cat_stats[cat]["total"] += 1

        results.append(dict(
            index=int(row["index"]), pred=pred, gt=gt,
            correct=hit, category=cat, gen_tokens=gen_tokens,
            response=text,
        ))

        if total % 50 == 0:
            acc_so_far = correct / total * 100
            vram_gb = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  [{total}/{len(df)}] acc={acc_so_far:.2f}%  "
                  f"avg_tokens={np.mean(gen_token_counts):.0f}  "
                  f"peak_vram={vram_gb:.2f}GB")

    elapsed = time.time() - t0
    acc = correct / max(total, 1) * 100
    vram_gb = (torch.cuda.max_memory_allocated() / 1024**3
               if torch.cuda.is_available() else 0)

    _print_summary("baseline", acc, correct, total, elapsed, vram_gb,
                   cat_stats, gen_token_counts)

    if args.output:
        _save_json(args.output, acc, total, correct, results, elapsed,
                   cat_stats, gen_token_counts, args, vram_gb)


# ═══════════════════════════════════════════════════════════════
# 子命令：extract
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def _extract_vit_only(model, processor, messages, blk_catcher):
    """只跑 ViT 前向（不跑 LLM 解码），用于快速提取中间层特征。"""
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v
              for k, v in inputs.items()}

    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")
    if pixel_values is not None and image_grid_thw is not None:
        model.model.visual(pixel_values, grid_thw=image_grid_thw)

    return blk_catcher.pop()


def cmd_extract(args):
    model, processor = load_model(args.model_path, args.device, args.dtype)
    df = load_mmstar(args.data_path, args.max_samples,
                     getattr(args, "sample_list", None))
    print(f"[data] {len(df)} samples loaded from MMStar")

    vision_model = model.model.visual
    layer = args.layer
    num_blocks = len(vision_model.blocks)
    assert 0 <= layer < num_blocks, \
        f"layer {layer} out of range [0, {num_blocks})"

    print(f"[extract] ViT depth={num_blocks}, "
          f"target layer={layer}, "
          f"hidden_size={vision_model.blocks[0].norm1.weight.shape[0]}")
    print(f"[extract] ViT-only mode (no LLM decoding)")

    blk_catcher = VisionBlockCatcher(vision_model.blocks, [layer])

    os.makedirs(args.out_dir, exist_ok=True)
    total = 0
    feat_shapes = []
    t0 = time.time()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="extract"):
        messages = build_messages(row)
        blk_feats = _extract_vit_only(model, processor, messages, blk_catcher)

        sid = str(row["index"])
        feat_tensor = blk_feats[f"blk{layer:02d}"]
        np.save(os.path.join(args.out_dir, f"{sid}.npy"),
                feat_tensor.numpy())

        feat_shapes.append(list(feat_tensor.shape))
        total += 1

        if total % 100 == 0:
            vram_gb = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  [{total}/{len(df)}]  "
                  f"feat_shape={feat_shapes[-1]}  "
                  f"peak_vram={vram_gb:.2f}GB")

    blk_catcher.close()

    elapsed = time.time() - t0
    vram_gb = (torch.cuda.max_memory_allocated() / 1024**3
               if torch.cuda.is_available() else 0)

    all_n = [s[0] for s in feat_shapes]
    hidden_dim = feat_shapes[0][1] if feat_shapes else 0

    print(f"\n{'=' * 65}")
    print(f"  [extract] Done! {total} features saved")
    print(f"  Time: {elapsed:.1f}s  ({elapsed/max(total,1):.2f}s/sample)")
    print(f"  Peak VRAM: {vram_gb:.2f}GB")
    print(f"  Feature shape: [N, {hidden_dim}]")
    print(f"  N (tokens) range: min={min(all_n)}, max={max(all_n)}, "
          f"mean={np.mean(all_n):.0f}, median={np.median(all_n):.0f}")
    print(f"  Output dir: {args.out_dir}")
    print(f"{'=' * 65}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        payload = {
            "mode": "extract (ViT-only)",
            "model": args.model_path,
            "layer": layer,
            "total": total,
            "elapsed_s": round(elapsed, 2),
            "peak_vram_gb": round(vram_gb, 2),
            "hidden_dim": hidden_dim,
            "token_count_stats": {
                "min": int(min(all_n)), "max": int(max(all_n)),
                "mean": round(float(np.mean(all_n)), 1),
                "median": round(float(np.median(all_n)), 1),
            },
            "feat_dir": args.out_dir,
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\n  → metadata saved to {args.output}")


# ═══════════════════════════════════════════════════════════════
# 子命令：replay
# ═══════════════════════════════════════════════════════════════

def cmd_replay(args):
    model, processor = load_model(args.model_path, args.device, args.dtype)
    df = load_mmstar(args.data_path, args.max_samples,
                     getattr(args, "sample_list", None))
    print(f"[data] {len(df)} samples loaded from MMStar")

    vision_model = model.model.visual
    layer = args.layer
    dtype = next(vision_model.parameters()).dtype
    device = next(vision_model.parameters()).device
    do_sample = not args.greedy

    print(f"[replay] layer={layer}, feat_dir={args.feat_dir}")

    correct = total = skipped = 0
    results = []
    cat_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    gen_token_counts = []
    t0 = time.time()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="replay"):
        sid = str(row["index"])
        feat_path = os.path.join(args.feat_dir, f"{sid}.npy")
        if not os.path.exists(feat_path):
            skipped += 1
            continue

        saved_blk = torch.from_numpy(np.load(feat_path))

        messages = build_messages(row)
        with replay_hooks(vision_model, saved_blk, layer, device, dtype):
            text, gen_tokens, _ = generate_answer(
                model, processor, messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repetition_penalty=args.repetition_penalty,
                presence_penalty=args.presence_penalty,
                do_sample=do_sample,
            )
        gen_token_counts.append(gen_tokens)

        pred = extract_answer(text)
        gt = row["answer"]
        hit = pred == gt
        correct += int(hit)
        total += 1

        cat = row.get("category", "unknown")
        cat_stats[cat]["correct"] += int(hit)
        cat_stats[cat]["total"] += 1

        results.append(dict(
            index=int(row["index"]), pred=pred, gt=gt,
            correct=hit, category=cat, gen_tokens=gen_tokens,
            response=text,
        ))

        if total % 50 == 0:
            acc_so_far = correct / total * 100
            vram_gb = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  [{total}/{len(df)}] acc={acc_so_far:.2f}%  "
                  f"avg_tokens={np.mean(gen_token_counts):.0f}  "
                  f"peak_vram={vram_gb:.2f}GB")

    elapsed = time.time() - t0
    acc = correct / max(total, 1) * 100
    vram_gb = (torch.cuda.max_memory_allocated() / 1024**3
               if torch.cuda.is_available() else 0)

    _print_summary("replay", acc, correct, total, elapsed, vram_gb,
                   cat_stats, gen_token_counts,
                   layer=layer, skipped=skipped)

    if args.output:
        _save_json(args.output, acc, total, correct, results, elapsed,
                   cat_stats, gen_token_counts, args, vram_gb,
                   replay_layer=layer, feat_dir=args.feat_dir,
                   skipped=skipped)


# ═══════════════════════════════════════════════════════════════
# 子命令：compare（baseline vs replay 一致性对比）
# ═══════════════════════════════════════════════════════════════

def cmd_compare(args):
    """随机选 N 个样本，baseline 与 replay 均用 greedy 推理，逐样本对比。"""
    model, processor = load_model(args.model_path, args.device, args.dtype)
    df = load_mmstar(args.data_path)
    print(f"[compare] MMStar total: {len(df)} samples")

    rng = np.random.RandomState(args.seed)
    n = min(args.num_samples, len(df))
    sel = sorted(rng.choice(len(df), size=n, replace=False))
    df_sub = df.iloc[sel].reset_index(drop=True)
    print(f"[compare] randomly selected {n} samples (seed={args.seed})")

    vision_model = model.model.visual
    layer = args.layer
    dtype_vit = next(vision_model.parameters()).dtype
    device_vit = next(vision_model.parameters()).device

    print(f"[compare] layer={layer}, feat_dir={args.feat_dir}")
    print(f"[compare] greedy decoding, max_new_tokens={args.max_new_tokens}")
    print()

    records = []
    t0 = time.time()

    for i, (_, row) in enumerate(tqdm(df_sub.iterrows(),
                                       total=len(df_sub), desc="compare")):
        sid = str(row["index"])
        gt = row["answer"]
        cat = row.get("category", "unknown")
        messages = build_messages(row)

        # ── baseline: 原始图片 → 完整 ViT → LLM (greedy) ──
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        text_base, tok_base, _ = generate_answer(
            model, processor, messages,
            max_new_tokens=args.max_new_tokens,
            do_sample=False, presence_penalty=0.0,
        )
        pred_base = extract_answer(text_base)

        # ── replay: 特征替换 ViT block[layer] → LLM (greedy) ──
        feat_path = os.path.join(args.feat_dir, f"{sid}.npy")
        if not os.path.exists(feat_path):
            print(f"  [SKIP] index={sid}: feature file not found")
            continue

        saved_blk = torch.from_numpy(np.load(feat_path))
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        with replay_hooks(vision_model, saved_blk, layer, device_vit, dtype_vit):
            text_replay, tok_replay, _ = generate_answer(
                model, processor, messages,
                max_new_tokens=args.max_new_tokens,
                do_sample=False, presence_penalty=0.0,
            )
        pred_replay = extract_answer(text_replay)

        pred_match = pred_base == pred_replay
        text_match = text_base == text_replay

        records.append(dict(
            index=int(row["index"]), category=cat, gt=gt,
            pred_base=pred_base, pred_replay=pred_replay,
            pred_match=pred_match, text_match=text_match,
            base_correct=(pred_base == gt),
            replay_correct=(pred_replay == gt),
            tok_base=tok_base, tok_replay=tok_replay,
            response_base=text_base, response_replay=text_replay,
        ))

    elapsed = time.time() - t0
    vram_gb = (torch.cuda.max_memory_allocated() / 1024**3
               if torch.cuda.is_available() else 0)

    n_total = len(records)
    n_pred_match = sum(r["pred_match"] for r in records)
    n_text_match = sum(r["text_match"] for r in records)
    n_base_ok = sum(r["base_correct"] for r in records)
    n_replay_ok = sum(r["replay_correct"] for r in records)

    # ── 表格输出 ──
    print(f"\n{'=' * 85}")
    print(f"  [compare] Baseline vs Replay — {n_total} samples, layer={layer}")
    print(f"  Decoding: greedy (deterministic)")
    print(f"  Time: {elapsed:.1f}s  Peak VRAM: {vram_gb:.2f}GB")
    print(f"{'=' * 85}")

    hdr = (f"{'idx':>6} | {'GT':>2} | {'Base':>4} | {'Rply':>4} | "
           f"{'Pred':>4} | {'Text':>4} | {'TokB':>5} | {'TokR':>5} | Category")
    print(hdr)
    print("-" * 85)

    for r in records:
        pm = "==" if r["pred_match"] else "!="
        tm = "==" if r["text_match"] else "!="
        print(f"{r['index']:>6} | {r['gt']:>2} | "
              f"{r['pred_base']:>4} | {r['pred_replay']:>4} | "
              f"{pm:>4} | {tm:>4} | "
              f"{r['tok_base']:>5} | {r['tok_replay']:>5} | {r['category']}")

    # ── 不一致样本详情 ──
    mismatches = [r for r in records if not r["text_match"]]
    if mismatches:
        print(f"\n{'─' * 85}")
        print(f"  Text-level mismatches ({len(mismatches)}/{n_total}):")
        for r in mismatches:
            print(f"\n  ── index={r['index']}  GT={r['gt']}  "
                  f"category={r['category']} ──")
            print(f"  BASE   pred={r['pred_base']}  tokens={r['tok_base']}")
            resp_b = r["response_base"].replace("\n", " ")[:300]
            print(f"    {resp_b}")
            print(f"  REPLAY pred={r['pred_replay']}  tokens={r['tok_replay']}")
            resp_r = r["response_replay"].replace("\n", " ")[:300]
            print(f"    {resp_r}")

    print(f"\n{'=' * 85}")
    print(f"  Summary:")
    pct = lambda a, b: a / max(b, 1) * 100
    print(f"    Answer match : {n_pred_match}/{n_total}"
          f"  ({pct(n_pred_match, n_total):.1f}%)")
    print(f"    Text   match : {n_text_match}/{n_total}"
          f"  ({pct(n_text_match, n_total):.1f}%)")
    print(f"    Base    acc   : {n_base_ok}/{n_total}"
          f"  ({pct(n_base_ok, n_total):.1f}%)")
    print(f"    Replay  acc   : {n_replay_ok}/{n_total}"
          f"  ({pct(n_replay_ok, n_total):.1f}%)")
    print(f"{'=' * 85}")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        payload = {
            "mode": "compare",
            "model": args.model_path,
            "layer": layer,
            "num_samples": n_total,
            "decoding": "greedy",
            "seed": args.seed,
            "elapsed_s": round(elapsed, 2),
            "peak_vram_gb": round(vram_gb, 2),
            "answer_match": f"{n_pred_match}/{n_total}",
            "text_match": f"{n_text_match}/{n_total}",
            "base_accuracy": round(pct(n_base_ok, n_total), 2),
            "replay_accuracy": round(pct(n_replay_ok, n_total), 2),
            "results": records,
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\n  → results saved to {args.output}")


# ═══════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════

def _print_summary(tag, acc, correct, total, elapsed, vram_gb,
                   cat_stats, gen_token_counts, **extra):
    tok_arr = np.array(gen_token_counts) if gen_token_counts else np.array([0])
    print(f"\n{'=' * 65}")
    print(f"  [{tag}] MMStar Accuracy: {acc:.2f}% ({correct}/{total})")
    print(f"  Time: {elapsed:.1f}s  ({elapsed/max(total,1):.2f}s/sample)")
    print(f"  Peak VRAM: {vram_gb:.2f}GB")
    for k, v in extra.items():
        print(f"  {k}: {v}")
    print(f"\n  Per-category accuracy:")
    for cat in sorted(cat_stats):
        s = cat_stats[cat]
        cat_acc = s["correct"] / max(s["total"], 1) * 100
        print(f"    {cat:30s}: {cat_acc:6.2f}%  ({s['correct']}/{s['total']})")
    print(f"\n  Token stats: min={int(tok_arr.min())}  "
          f"median={np.median(tok_arr):.0f}  mean={tok_arr.mean():.0f}  "
          f"max={int(tok_arr.max())}  total={int(tok_arr.sum())}")
    print(f"{'=' * 65}")


def _save_json(path, acc, total, correct, results, elapsed,
               cat_stats, gen_token_counts, args, vram_gb, **extra):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    cat_acc = {}
    for cat in sorted(cat_stats):
        s = cat_stats[cat]
        cat_acc[cat] = round(s["correct"] / max(s["total"], 1) * 100, 2)

    tok_arr = np.array(gen_token_counts) if gen_token_counts else np.array([0])
    token_stats = {
        "min": int(tok_arr.min()), "max": int(tok_arr.max()),
        "mean": round(float(tok_arr.mean()), 1),
        "median": round(float(np.median(tok_arr)), 1),
        "total": int(tok_arr.sum()),
    }

    payload = {
        "model": args.model_path,
        "accuracy": round(acc, 2),
        "total": total,
        "correct": correct,
        "elapsed_s": round(elapsed, 2),
        "peak_vram_gb": round(vram_gb, 2),
        "config": {
            "seed": getattr(args, "seed", None),
            "do_sample": not args.greedy,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "presence_penalty": args.presence_penalty,
            "max_new_tokens": args.max_new_tokens,
            "dtype": args.dtype,
        },
        "category_accuracy": cat_acc,
        "token_stats": token_stats,
    }
    payload.update(extra)
    payload["results"] = results

    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"\n  → results saved to {path}")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def _add_common_args(parser):
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--data_path", required=True,
                        help="MMStar parquet 文件路径")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--sample_list", default=None,
                        help="采样 list 文件路径（按 index 过滤）")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--presence_penalty", type=float, default=1.5)
    parser.add_argument("--greedy", action="store_true",
                        help="使用 greedy 解码（覆盖 sampling 参数）")


def build_parser():
    ap = argparse.ArgumentParser(
        "Qwen3-VL-8B-Instruct：MMStar 分层抽取 / 分层回放 VQA"
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # --- baseline ---
    pb = sub.add_parser("baseline", help="标准 MMStar 评测（精度基线）")
    _add_common_args(pb)
    pb.add_argument("--output", default=None)

    # --- extract ---
    pe = sub.add_parser("extract",
                        help="hook 抽取 ViT 中间层特征 + 同步评测")
    _add_common_args(pe)
    pe.add_argument("--layer", type=int, default=8,
                    help="ViT block 切分索引（0-based，8B 共27层，默认8=首个 deepstack 层）")
    pe.add_argument("--out_dir", required=True,
                    help="特征 .npy 输出目录")
    pe.add_argument("--output", default=None)

    # --- replay ---
    pr = sub.add_parser("replay",
                        help="从保存/重建特征回放推理")
    _add_common_args(pr)
    pr.add_argument("--layer", type=int, default=8,
                    help="ViT block 切分索引（须与 extract 一致）")
    pr.add_argument("--feat_dir", required=True,
                    help="extract 输出或 codec 重建的特征目录")
    pr.add_argument("--output", default=None)

    # --- compare ---
    pc = sub.add_parser("compare",
                        help="随机选样本，baseline vs replay greedy 对比")
    _add_common_args(pc)
    pc.add_argument("--layer", type=int, default=8)
    pc.add_argument("--feat_dir", required=True,
                    help="已提取特征目录")
    pc.add_argument("--num_samples", type=int, default=20,
                    help="随机抽取样本数（默认 20）")
    pc.add_argument("--output", default=None)

    return ap


def main():
    args = build_parser().parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.cmd == "baseline":
        cmd_baseline(args)
    elif args.cmd == "extract":
        cmd_extract(args)
    elif args.cmd == "replay":
        cmd_replay(args)
    elif args.cmd == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()

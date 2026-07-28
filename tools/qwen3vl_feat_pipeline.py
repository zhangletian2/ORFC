# -*- coding: utf-8 -*-
"""
Qwen3-VL-4B-Thinking MMBench 管线（分层抽取 / 分层回放 VQA）：

- baseline：直接跑 MMBench 评测作为精度基线。
- extract ：在 ViT 指定层（默认 block[5]）注册 forward hook，
            保存该层 hidden_states [N, 1024]。
- replay  ：加载保存的中间层特征（或 codec 重建特征），用 hook
            替换 ViT block 输出，后续的 DeepStack 和 block 均从
            重建特征自然重算 → merger → LLM generate。

设计意图：将 ViT 在 block[5] 处拆分为前后两半，前半 blocks 0-5
的输出经 codec 编解码后，送入后半 blocks 6-23 + DeepStack + merger
完成端到端推理。

Qwen3-VL-4B-Thinking 视觉塔参数：
  depth=24, hidden_size=1024, patch_size=16, spatial_merge_size=2
  deepstack_visual_indexes=[5, 11, 17]
  merger → out_hidden_size=2560

依赖：transformers>=4.57, torch>=2.1, qwen-vl-utils, datasets, Pillow
"""

import os
import sys
import time
import json
import argparse
import re
from io import BytesIO
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm


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


def load_mmbench(data_dir, split="validation", config_name=None,
                 data_subdir="en", max_samples=None, sample_list=None):
    from datasets import load_dataset

    kwargs = {}
    if config_name:
        kwargs["name"] = config_name
    if data_subdir:
        kwargs["data_dir"] = data_subdir
    ds = load_dataset(data_dir, split=split, **kwargs)

    if sample_list:
        index_set = _load_sample_list(sample_list)
        keep = [i for i, s in enumerate(ds) if s["index"] in index_set]
        ds = ds.select(keep)
        print(f"[sample_list] filtered {len(keep)} samples "
              f"from {sample_list}")
    elif max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds


def _load_sample_list(path):
    """读取采样 list 文件，返回 index 集合。

    格式：每行第一列为 sample index（int），tab 分隔。
    """
    indices = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            indices.add(int(line.split("\t")[0]))
    return indices


# ═══════════════════════════════════════════════════════════════
# MMBench prompt 构造 / 答案解析
# ═══════════════════════════════════════════════════════════════

def build_prompt(sample):
    """将 MMBench 样本转为 Qwen3-VL chat messages 格式。"""
    question = sample["question"]
    hint = sample.get("hint", None) or ""
    options = []
    for key in ("A", "B", "C", "D"):
        val = sample.get(key)
        if val:
            options.append(f"{key}. {val}")

    text_parts = []
    if hint and hint.strip():
        text_parts.append(f"Hint: {hint.strip()}")
    text_parts.append(question)
    text_parts.extend(options)
    text_parts.append(
        "Answer with the option's letter from the given choices directly."
    )
    text = "\n".join(text_parts)

    img = sample["image"]
    if not isinstance(img, Image.Image):
        img = Image.open(BytesIO(img)).convert("RGB")
    elif img.mode != "RGB":
        img = img.convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": text},
            ],
        }
    ]
    return messages


def _inject_no_think(messages):
    """在 user message 末尾追加 /no_think 标记以关闭 thinking。"""
    for msg in messages:
        if msg.get("role") == "user":
            for item in msg.get("content", []):
                if isinstance(item, dict) and item.get("type") == "text":
                    if "/no_think" not in item["text"]:
                        item["text"] = "/no_think\n" + item["text"]
                    return


_ANSWER_PHRASES = [
    "the answer is", "answer is", "the correct answer is",
    "correct answer is", "the best answer is", "best answer is",
    "the correct option is", "correct option is",
    "the best option is", "best option is",
    "the choice is", "choice is", "the correct choice is",
    "correct choice is", "i choose", "i select", "i pick",
    "my answer is", "my choice is",
    "答案是", "答案为", "选",
]

_FMT_PRIO = {
    "start": 10, "end": 9, "phrase": 7, "parentheses": 6,
    "period": 5, "colon": 4, "right_paren": 3, "space": 2, "fallback": 0,
}


def extract_mcq_answer(response, choices=None):
    """从模型输出中提取选择题答案（优先级排序的鲁棒正则匹配）。

    兼容 Thinking 模型输出（自动剥离 <think>...</think>），
    覆盖 10+ 种常见答案格式，与 lmms-eval mcq_extract.py 对齐。
    """
    if not response or not response.strip():
        return ""
    if "</think>" in response:
        response = response.split("</think>")[-1]

    choices = choices or ["A", "B", "C", "D"]
    text = response.strip()
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
# CircularEval 工具
# ═══════════════════════════════════════════════════════════════

def _get_valid_options(sample):
    """返回样本中有效选项的 (label, value) 列表，过滤 NaN。"""
    opts = []
    for key in ("A", "B", "C", "D"):
        val = sample.get(key)
        if val is not None and str(val).strip() \
                and str(val).strip().lower() != "nan":
            opts.append((key, str(val)))
    return opts


def _build_circular_pass(sample, rotation):
    """生成第 rotation 轮循环排列的 (messages, gt_letter, valid_labels)。

    CircularEval 策略：将选项值做循环移位，使每一轮的正确答案标签
    不同，k 轮全部答对才计为 hit=1（排除随机猜测）。
    """
    opts = _get_valid_options(sample)
    k = len(opts)
    labels = [o[0] for o in opts]
    values = [o[1] for o in opts]

    rotated = values[rotation:] + values[:rotation]

    mod = dict(sample)
    for key in ("A", "B", "C", "D"):
        mod[key] = None
    for i, label in enumerate(labels):
        mod[label] = rotated[i]

    ans_idx = labels.index(sample["answer"])
    new_gt = labels[(ans_idx - rotation) % k]

    return build_prompt(mod), new_gt, labels


# ═══════════════════════════════════════════════════════════════
# Hook 工具：抽取 / 回放
# ═══════════════════════════════════════════════════════════════

class VisionBlockCatcher:
    """在指定 ViT block 上注册 forward hook，抓取输出 hidden_states。"""

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


class DeepStackCatcher:
    """在指定 deepstack merger 上注册 forward hook，抓取输出。"""

    def __init__(self, merger_list, merger_indices):
        self.buf = {}
        self._handles = []
        for idx in merger_indices:
            key = f"ds{idx}"

            def _hook(_mod, _inp, out, _key=key):
                self.buf[_key] = out.detach().cpu().float()

            self._handles.append(merger_list[idx].register_forward_hook(_hook))

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

    注意：不替换 deepstack merger 的输出。当切分点恰好是 deepstack 层
    （如 layer=5 对应 deepstack_visual_indexes[0]）时，deepstack merger
    会从替换后的 hidden_states 自然重算，确保下游链路完整反映 codec 影响。
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
def generate_answer(model, processor, messages, max_new_tokens=512,
                    no_think=False):
    """对单条 messages 做 generate，返回解码后的文本。"""
    if no_think:
        _inject_no_think(messages)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v
              for k, v in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    input_len = inputs["input_ids"].shape[1]
    gen_ids = output_ids[0, input_len:]
    text = processor.decode(gen_ids, skip_special_tokens=True)
    return text, inputs


# ═══════════════════════════════════════════════════════════════
# 子命令：baseline
# ═══════════════════════════════════════════════════════════════

def cmd_baseline(args):
    model, processor = load_model(args.model_path, args.device, args.dtype)
    ds = load_mmbench(args.data_dir, args.split, args.config_name,
                      args.data_subdir, args.max_samples,
                      getattr(args, "sample_list", None))

    correct = total = 0
    results = []
    t0 = time.time()
    circular = getattr(args, "circular", False)

    for sample in tqdm(ds, desc="baseline"):
        if circular:
            opts = _get_valid_options(sample)
            k = len(opts)
            all_ok = True
            pass_details = []
            for r in range(k):
                msgs, gt, valid = _build_circular_pass(sample, r)
                text, _ = generate_answer(
                    model, processor, msgs,
                    max_new_tokens=args.max_new_tokens,
                    no_think=args.no_think,
                )
                pred = extract_mcq_answer(text, valid)
                pass_details.append(dict(
                    rotation=r, pred=pred, gt=gt, raw=text,
                ))
                if pred != gt:
                    all_ok = False
                    break
            correct += int(all_ok)
            total += 1
            results.append(dict(
                index=sample["index"], hit=int(all_ok),
                passes=pass_details,
            ))
        else:
            messages = build_prompt(sample)
            text, _ = generate_answer(
                model, processor, messages,
                max_new_tokens=args.max_new_tokens,
                no_think=args.no_think,
            )
            pred = extract_mcq_answer(text)
            gt = sample["answer"]
            ok = pred == gt
            correct += int(ok)
            total += 1
            results.append(dict(
                index=sample["index"], pred=pred, gt=gt,
                raw=text, correct=ok,
            ))

    elapsed = time.time() - t0
    acc = correct / max(total, 1) * 100
    mode = "CircularEval" if circular else "single-pass"
    vram_gb = (torch.cuda.max_memory_allocated() / 1024**3
               if torch.cuda.is_available() else 0)
    print(f"\n[baseline] Acc={acc:.2f}%  ({correct}/{total})  "
          f"mode={mode}  time={elapsed:.1f}s  "
          f"think={'off' if args.no_think else 'on'}  "
          f"peak_vram={vram_gb:.2f}GB")

    if args.output:
        _save_json(args.output, acc, total, correct, results, elapsed,
                   eval_mode=mode, peak_vram_gb=round(vram_gb, 2))


# ═══════════════════════════════════════════════════════════════
# 子命令：extract
# ═══════════════════════════════════════════════════════════════

def cmd_extract(args):
    model, processor = load_model(args.model_path, args.device, args.dtype)
    ds = load_mmbench(args.data_dir, args.split, args.config_name,
                      args.data_subdir, args.max_samples,
                      getattr(args, "sample_list", None))

    vision_model = model.model.visual
    layer = args.layer
    num_blocks = len(vision_model.blocks)
    assert 0 <= layer < num_blocks, \
        f"layer {layer} out of range [0, {num_blocks})"

    blk_catcher = VisionBlockCatcher(vision_model.blocks, [layer])

    os.makedirs(args.out_dir, exist_ok=True)
    correct = total = 0
    results = []
    t0 = time.time()

    for sample in tqdm(ds, desc="extract"):
        messages = build_prompt(sample)
        answer_text, inputs = generate_answer(
            model, processor, messages,
            max_new_tokens=args.max_new_tokens,
            no_think=args.no_think,
        )

        blk_feats = blk_catcher.pop()
        sid = str(sample["index"])
        np.save(
            os.path.join(args.out_dir, f"{sid}.npy"),
            blk_feats[f"blk{layer:02d}"].numpy(),
        )

        pred = extract_mcq_answer(answer_text)
        gt = sample["answer"]
        ok = pred == gt
        correct += int(ok)
        total += 1
        results.append(dict(
            index=sample["index"], pred=pred, gt=gt,
            raw=answer_text, correct=ok,
        ))

    blk_catcher.close()

    elapsed = time.time() - t0
    acc = correct / max(total, 1) * 100
    vram_gb = (torch.cuda.max_memory_allocated() / 1024**3
               if torch.cuda.is_available() else 0)
    print(f"\n[extract] Acc={acc:.2f}%  ({correct}/{total})  "
          f"layer={layer}  time={elapsed:.1f}s  "
          f"peak_vram={vram_gb:.2f}GB")
    print(f"[extract] Features → {args.out_dir}")

    if args.output:
        _save_json(args.output, acc, total, correct, results, elapsed,
                   peak_vram_gb=round(vram_gb, 2))


# ═══════════════════════════════════════════════════════════════
# 子命令：replay
# ═══════════════════════════════════════════════════════════════

def cmd_replay(args):
    model, processor = load_model(args.model_path, args.device, args.dtype)
    ds = load_mmbench(args.data_dir, args.split, args.config_name,
                      args.data_subdir, args.max_samples,
                      getattr(args, "sample_list", None))

    vision_model = model.model.visual
    layer = args.layer
    dtype = next(vision_model.parameters()).dtype
    device = next(vision_model.parameters()).device
    circular = getattr(args, "circular", False)

    correct = total = skipped = 0
    results = []
    t0 = time.time()

    for sample in tqdm(ds, desc="replay"):
        sid = str(sample["index"])
        feat_path = os.path.join(args.feat_dir, f"{sid}.npy")
        if not os.path.exists(feat_path):
            skipped += 1
            continue

        saved_blk = torch.from_numpy(np.load(feat_path))

        if circular:
            opts = _get_valid_options(sample)
            k = len(opts)
            all_ok = True
            pass_details = []
            with replay_hooks(vision_model, saved_blk,
                              layer, device, dtype):
                for r in range(k):
                    msgs, gt, valid = _build_circular_pass(sample, r)
                    text, _ = generate_answer(
                        model, processor, msgs,
                        max_new_tokens=args.max_new_tokens,
                        no_think=args.no_think,
                    )
                    pred = extract_mcq_answer(text, valid)
                    pass_details.append(dict(
                        rotation=r, pred=pred, gt=gt, raw=text,
                    ))
                    if pred != gt:
                        all_ok = False
                        break
            correct += int(all_ok)
            total += 1
            results.append(dict(
                index=sid, hit=int(all_ok), passes=pass_details,
            ))
        else:
            messages = build_prompt(sample)
            with replay_hooks(vision_model, saved_blk,
                              layer, device, dtype):
                text, _ = generate_answer(
                    model, processor, messages,
                    max_new_tokens=args.max_new_tokens,
                    no_think=args.no_think,
                )
            pred = extract_mcq_answer(text)
            gt = sample["answer"]
            ok = pred == gt
            correct += int(ok)
            total += 1
            results.append(dict(
                index=sid, pred=pred, gt=gt, raw=text, correct=ok,
            ))

    elapsed = time.time() - t0
    acc = correct / max(total, 1) * 100
    mode = "CircularEval" if circular else "single-pass"
    vram_gb = (torch.cuda.max_memory_allocated() / 1024**3
               if torch.cuda.is_available() else 0)
    print(f"\n[replay] Acc={acc:.2f}%  ({correct}/{total})  "
          f"layer={layer}  mode={mode}  skipped={skipped}  "
          f"time={elapsed:.1f}s  peak_vram={vram_gb:.2f}GB")

    if args.output:
        _save_json(args.output, acc, total, correct, results, elapsed,
                   eval_mode=mode, peak_vram_gb=round(vram_gb, 2))


# ═══════════════════════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════════════════════

def _save_json(path, acc, total, correct, results, elapsed, **extra):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = dict(
        accuracy=acc, total=total, correct=correct,
        elapsed_s=round(elapsed, 2),
    )
    payload.update(extra)
    payload["results"] = results
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"  → results saved to {path}")


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def build_parser():
    ap = argparse.ArgumentParser(
        "Qwen3-VL-4B-Thinking：MMBench 分层抽取 / 分层回放 VQA"
    )
    ap.add_argument("--model_path", required=True,
                    help="模型本地路径或 HuggingFace ID")
    ap.add_argument("--data_dir", required=True,
                    help="MMBench 数据集目录")
    ap.add_argument("--split", default="validation",
                    help="数据集 split（validation / test）")
    ap.add_argument("--config_name", default=None,
                    help="datasets config name（多语言时指定）")
    ap.add_argument("--data_subdir", default="en",
                    help="数据集子目录（en / cn / cc）")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bf16",
                    choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--max_new_tokens", type=int, default=2048,
                    help="生成最大 token 数（VLMEvalKit Qwen-VL 默认 2048）")
    ap.add_argument("--max_samples", type=int, default=None,
                    help="最多评测样本数（调试用）")
    ap.add_argument("--sample_list", default=None,
                    help="采样 list 文件路径（按 index 过滤，优先于 max_samples）")
    ap.add_argument("--no_think", action="store_true",
                    help="关闭 thinking 模式（直接输出答案）")

    sub = ap.add_subparsers(dest="cmd", required=True)

    # --- baseline ---
    pb = sub.add_parser("baseline", help="标准 MMBench 评测")
    pb.add_argument("--circular", action="store_true",
                    help="启用 CircularEval（与官方排行榜对齐）")
    pb.add_argument("--output", default=None,
                    help="JSON 结果输出路径")

    # --- extract ---
    pe = sub.add_parser("extract",
                        help="hook 抽取 ViT 中间层特征")
    pe.add_argument("--layer", type=int, default=5,
                    help="ViT block 切分索引（0-based，共24层，默认5=前半最后一层）")
    pe.add_argument("--out_dir", required=True,
                    help="特征 .npy 输出目录")
    pe.add_argument("--output", default=None,
                    help="JSON 结果输出路径")

    # --- replay ---
    pr = sub.add_parser("replay",
                        help="从保存/重建特征回放推理")
    pr.add_argument("--layer", type=int, default=5,
                    help="ViT block 切分索引（须与 extract 一致）")
    pr.add_argument("--feat_dir", required=True,
                    help="extract 输出的特征目录")
    pr.add_argument("--circular", action="store_true",
                    help="启用 CircularEval（与官方排行榜对齐）")
    pr.add_argument("--output", default=None,
                    help="JSON 结果输出路径")

    return ap


def main():
    args = build_parser().parse_args()
    if args.cmd == "baseline":
        cmd_baseline(args)
    elif args.cmd == "extract":
        cmd_extract(args)
    elif args.cmd == "replay":
        cmd_replay(args)


if __name__ == "__main__":
    main()

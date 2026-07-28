"""
VTM + SigLIP2 Retrieval 验证脚本

每层取 2 个样本，在所有 QP 下做完整 编码→解码→回放→检索评估，用于：
  1. 验证 VTM + replay 流水线端到端正确性（embedding CosSim + 检索指标）
  2. 检验 QP 取值是否产生合理 BPFP 范围（覆盖 0.05 ~ min(0.75, 无损)）

Usage:
  conda activate siglip_codec
  cd /data4/workspace/zlt/featcodec/ORFC/coding/vtm_baseline
  python verify_vtm_siglip2_ret.py
"""

import os, sys, subprocess, shutil, threading, json
import numpy as np
from pathlib import Path
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor, as_completed

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "tools"))

# ─────────────── 配置 ───────────────
FEAT_ROOT = PROJECT_ROOT / "features" / "coco_ret" / "siglip2_so400m"
CAPTION_JSON = PROJECT_ROOT / "utils" / "coco_selected_caption500.json"
COCO_ROOT = Path("/data4/workspace/zlt/CAPO_ret/dataset/coco2014")
MODEL_ID = "google/siglip2-so400m-patch14-224"

VTM_ENCODER = str(SCRIPT_DIR / "EncoderAppStatic")
VTM_DECODER = str(SCRIPT_DIR / "DecoderAppStatic")
VTM_CFG = str(SCRIPT_DIR / "encoder_intra_vtm.cfg")
BIT_DEPTH = 10

LAYERS = ["blk07", "blk15", "blk23"]
QPS_MAP = {
    "blk07": [22, 25, 27, 30, 32],
    "blk15": [0, 2, 5, 7, 10, 12],
    "blk23": [0, 2, 5, 7, 10, 12],
}
N_SAMPLES = 2
DEVICE = os.environ.get("CUDA_DEVICE", "cuda")
VTM_WORKERS = 10


# ─────────────── VTM 编解码 ───────────────

def quantize_linear(x, bit_depth):
    x = x.astype(np.float32)
    xmin, xmax = float(x.min()), float(x.max())
    if xmax <= xmin:
        scale = 1.0
    else:
        scale = ((1 << bit_depth) - 1) / (xmax - xmin)
    q = np.round((x - xmin) * scale).astype(np.uint16 if bit_depth > 8 else np.uint8)
    return q, {"min": xmin, "max": xmax, "bit_depth": bit_depth}


def dequantize_linear(q, meta):
    q = q.astype(np.float32)
    B = int(meta["bit_depth"])
    xmin, xmax = float(meta["min"]), float(meta["max"])
    if xmax <= xmin:
        return np.full_like(q, xmin, dtype=np.float32)
    scale = ((1 << B) - 1) / (xmax - xmin)
    return (q / scale + xmin).astype(np.float32)


def vtm_encode_decode(feat_2d, qp, tmp_dir):
    H, W = feat_2d.shape
    q, meta = quantize_linear(feat_2d, BIT_DEPTH)

    uid = f"vret_{os.getpid()}_{threading.get_ident()}_{qp}"
    yuv_path = os.path.join(tmp_dir, f"{uid}.y")
    bitstream = os.path.join(tmp_dir, f"{uid}.vvc")
    dec_yuv = os.path.join(tmp_dir, f"{uid}.dec.y")
    enc_log = os.path.join(tmp_dir, f"{uid}.enc.log")
    dec_log = os.path.join(tmp_dir, f"{uid}.dec.log")

    q.tofile(yuv_path)

    cmd_enc = [
        VTM_ENCODER, "-c", VTM_CFG,
        "-i", yuv_path, "-b", bitstream,
        f"--SourceWidth={W}", f"--SourceHeight={H}",
        "--FramesToBeEncoded=1", "--FrameRate=1",
        "--InputChromaFormat=400", "--ConformanceWindowMode=1",
        f"--InternalBitDepth={BIT_DEPTH}",
        f"--InputBitDepth={BIT_DEPTH}",
        f"--OutputBitDepth={BIT_DEPTH}",
        f"--QP={qp}",
    ]
    with open(enc_log, "w") as f:
        subprocess.run(cmd_enc, stdout=f, stderr=subprocess.STDOUT, check=True)

    cmd_dec = [VTM_DECODER, "-b", bitstream, "-o", dec_yuv]
    with open(dec_log, "w") as f:
        subprocess.run(cmd_dec, stdout=f, stderr=subprocess.STDOUT, check=True)

    dt = np.uint16 if BIT_DEPTH > 8 else np.uint8
    q_rec = np.fromfile(dec_yuv, dtype=dt).reshape(H, W)
    feat_rec = dequantize_linear(q_rec, meta)
    bs_bytes = os.path.getsize(bitstream)

    for p in [yuv_path, bitstream, dec_yuv, enc_log, dec_log]:
        try:
            os.unlink(p)
        except OSError:
            pass

    return feat_rec, bs_bytes


# ─────────────── 并行 VTM 任务 ───────────────

def vtm_task(args_tuple):
    layer, qp, idx, name, feat_path, tmp_dir = args_tuple
    feat_orig = np.load(feat_path)
    if feat_orig.ndim == 3 and feat_orig.shape[0] == 1:
        feat_orig = feat_orig[0]
    H, W = feat_orig.shape
    feat_rec, bs_bytes = vtm_encode_decode(feat_orig, qp, tmp_dir)
    bpfp = (bs_bytes * 8) / (H * W)
    feat_mse = float(np.mean((feat_orig - feat_rec) ** 2))
    cos_sim = float(np.dot(feat_orig.flatten(), feat_rec.flatten()) /
                    (np.linalg.norm(feat_orig) * np.linalg.norm(feat_rec) + 1e-12))
    rec_path = os.path.join(tmp_dir, f"{layer}_{qp}_{name}.npy")
    np.save(rec_path, feat_rec)
    return (layer, qp, idx, name, bpfp, feat_mse, cos_sim, rec_path)


# ─────────────── 检索评估工具 ───────────────

def build_retrieval_maps(all_meta):
    """构建 img2txt / txt2img 映射和 caption 列表。"""
    img2txt, txt2img, caption_list = {}, {}, []
    cap_idx = 0
    for i, e in enumerate(all_meta):
        gt = []
        for cap in e['caption']:
            caption_list.append(cap)
            txt2img[cap_idx] = i
            gt.append(cap_idx)
            cap_idx += 1
        img2txt[i] = gt
    return caption_list, img2txt, txt2img


def eval_retrieval(score_matrix, img2txt, txt2img):
    """I2T / T2I R@1, R@5, R@10。"""
    n_img, n_txt = score_matrix.shape

    i2t_r1 = i2t_r5 = i2t_r10 = 0
    for i in range(n_img):
        ranking = np.argsort(-score_matrix[i])
        gt_set = set(img2txt[i])
        for rank, idx in enumerate(ranking):
            if idx in gt_set:
                if rank < 1:
                    i2t_r1 += 1
                if rank < 5:
                    i2t_r5 += 1
                if rank < 10:
                    i2t_r10 += 1
                break

    t2i_r1 = t2i_r5 = t2i_r10 = 0
    for j in range(n_txt):
        ranking = np.argsort(-score_matrix[:, j])
        gt = txt2img[j]
        rank = int(np.where(ranking == gt)[0][0])
        if rank < 1:
            t2i_r1 += 1
        if rank < 5:
            t2i_r5 += 1
        if rank < 10:
            t2i_r10 += 1

    return {
        "i2t_R@1":  i2t_r1  / n_img * 100,
        "i2t_R@5":  i2t_r5  / n_img * 100,
        "i2t_R@10": i2t_r10 / n_img * 100,
        "t2i_R@1":  t2i_r1  / n_txt * 100,
        "t2i_R@5":  t2i_r5  / n_txt * 100,
        "t2i_R@10": t2i_r10 / n_txt * 100,
    }


def i2t_rank_for_image(score_row, gt_indices):
    """单张图片的 I2T 首个 GT caption 排名（0-based）。"""
    ranking = np.argsort(-score_row)
    gt_set = set(gt_indices)
    for rank, idx in enumerate(ranking):
        if idx in gt_set:
            return rank
    return len(ranking)


# ─────────────── 主流程 ───────────────

def main():
    print("=" * 90)
    print("  VTM + SigLIP2 Retrieval 验证脚本")
    print(f"  每层 {N_SAMPLES} 个样本 × 所有 QP + 全量检索评估")
    print(f"  模型: {MODEL_ID}")
    print(f"  特征: {FEAT_ROOT}")
    print("=" * 90)

    with open(CAPTION_JSON) as f:
        all_meta = json.load(f)

    n_images = len(all_meta)
    all_image_ids = [e["image_id"] for e in all_meta]
    test_indices = list(range(N_SAMPLES))
    sample_names = [all_image_ids[i] for i in test_indices]
    print(f"  数据集: {n_images} 张图片, {n_images * 5} 条 caption")
    print(f"  VTM 测试样本: {sample_names}")

    caption_list, img2txt, txt2img = build_retrieval_maps(all_meta)

    tmp_dir = str(SCRIPT_DIR / "_vtm_verify_ret_tmp")
    os.makedirs(tmp_dir, exist_ok=True)

    # ================================================================
    #  Phase 1: 并行 VTM 编解码（纯 CPU）
    # ================================================================
    print(f"\n[1/4] VTM 编解码 (workers={VTM_WORKERS})...")

    tasks = []
    for layer in LAYERS:
        qps = QPS_MAP[layer]
        for qp in qps:
            for i, name in enumerate(sample_names):
                feat_path = str(FEAT_ROOT / layer / f"{name}.npy")
                tasks.append((layer, qp, i, name, feat_path, tmp_dir))

    print(f"  总任务数: {len(tasks)} ({len(LAYERS)} layers × QPs × {N_SAMPLES} samples)")
    t0_all = perf_counter()

    vtm_results = {}
    completed = 0
    with ProcessPoolExecutor(max_workers=VTM_WORKERS) as executor:
        futures = {executor.submit(vtm_task, t): t for t in tasks}
        for future in as_completed(futures):
            completed += 1
            try:
                layer, qp, idx, name, bpfp, feat_mse, cos_sim, rec_path = future.result()
                vtm_results[(layer, qp, idx)] = {
                    'name': name, 'bpfp': bpfp, 'feat_mse': feat_mse,
                    'cos_sim': cos_sim, 'rec_path': rec_path,
                }
                if completed % 5 == 0 or completed == len(tasks):
                    print(f"  VTM 进度: {completed}/{len(tasks)}")
            except Exception as e:
                task_info = futures[future]
                print(f"  [ERR] {task_info[0]}/QP{task_info[1]}/{task_info[3]}: {e}")

    vtm_elapsed = perf_counter() - t0_all
    print(f"  VTM 编解码完成: {vtm_elapsed:.1f}s")

    # ================================================================
    #  Phase 2: 编码全部文本
    # ================================================================
    print(f"\n[2/4] 加载 SigLIP2 模型，编码文本...")

    import torch
    from siglip2_feat_pipeline import load_siglip2, continue_from_tokens
    from tqdm import tqdm

    model, processor = load_siglip2(MODEL_ID, DEVICE)
    model.float()
    vision_model = model.vision_model
    logit_scale = model.logit_scale.exp().item()

    all_txt_emb = []
    bs_txt = 128
    for i in tqdm(range(0, len(caption_list), bs_txt), desc="Encoding texts"):
        batch = caption_list[i:i + bs_txt]
        inputs = processor(
            text=batch, padding="max_length", max_length=64,
            truncation=True, return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            out = model.get_text_features(**inputs)
            emb = out.pooler_output if hasattr(out, 'pooler_output') else out
            emb = emb.float()
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        all_txt_emb.append(emb.cpu())
    txt_emb = torch.cat(all_txt_emb, dim=0)
    print(f"  文本嵌入: {txt_emb.shape}")

    # ================================================================
    #  Phase 3: 每层 Anchor + VTM replay → 全量检索
    # ================================================================
    print(f"\n[3/4] 逐层回放全量图片 + VTM 替换检索评估...")

    all_results = {}

    for layer in LAYERS:
        layer_idx = int(layer.replace("blk", ""))
        qps = QPS_MAP[layer]

        print(f"\n  ─── {layer} (layer_idx={layer_idx}) ───")

        # Anchor: 回放全部 500 张原始特征 → image embeddings
        print(f"    回放全部 {n_images} 张原始特征...")
        anchor_img_emb = []
        for e in tqdm(all_meta, desc=f"    Anchor {layer}", leave=False):
            feat = np.load(str(FEAT_ROOT / layer / f"{e['image_id']}.npy"))
            tok = torch.from_numpy(feat).unsqueeze(0).to(DEVICE, dtype=torch.float32)
            with torch.no_grad():
                emb = continue_from_tokens(vision_model, tok, layer_idx)
                emb = emb.float()
                emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            anchor_img_emb.append(emb.cpu())
        anchor_img_emb = torch.cat(anchor_img_emb, dim=0)

        score_anchor = logit_scale * (anchor_img_emb @ txt_emb.t())
        score_anchor_np = score_anchor.numpy()
        ret_anchor = eval_retrieval(score_anchor_np, img2txt, txt2img)

        anchor_ranks = []
        for ti in test_indices:
            r = i2t_rank_for_image(score_anchor_np[ti], img2txt[ti])
            anchor_ranks.append(r)

        print(f"    Anchor: i2t R@1={ret_anchor['i2t_R@1']:.2f}%  "
              f"R@5={ret_anchor['i2t_R@5']:.2f}%  R@10={ret_anchor['i2t_R@10']:.2f}%")
        print(f"            t2i R@1={ret_anchor['t2i_R@1']:.2f}%  "
              f"R@5={ret_anchor['t2i_R@5']:.2f}%  R@10={ret_anchor['t2i_R@10']:.2f}%")
        for si, ti in enumerate(test_indices):
            print(f"      [{sample_names[si]}] I2T rank={anchor_ranks[si]}")

        all_results[(layer, "anchor")] = {
            'ret': ret_anchor, 'ranks': anchor_ranks,
        }

        # 每个 QP: 替换 test 样本 embedding → 重新检索
        for qp in qps:
            bpfp_list, mse_list, feat_cos_list, emb_cos_list = [], [], [], []
            img_emb_replaced = anchor_img_emb.clone()
            vtm_ranks = []

            for si, (gi, name) in enumerate(zip(test_indices, sample_names)):
                key = (layer, qp, si)
                if key not in vtm_results:
                    continue
                r = vtm_results[key]
                bpfp_list.append(r['bpfp'])
                mse_list.append(r['feat_mse'])
                feat_cos_list.append(r['cos_sim'])

                feat_rec = np.load(r['rec_path'])
                tok = torch.from_numpy(feat_rec).unsqueeze(0).to(DEVICE, dtype=torch.float32)
                with torch.no_grad():
                    emb_rec = continue_from_tokens(vision_model, tok, layer_idx)
                    emb_rec = emb_rec.float()
                    emb_rec = emb_rec / emb_rec.norm(dim=-1, keepdim=True).clamp(min=1e-12)

                emb_cs = torch.nn.functional.cosine_similarity(
                    anchor_img_emb[gi:gi + 1], emb_rec.cpu(), dim=-1).item()
                emb_cos_list.append(emb_cs)

                img_emb_replaced[gi] = emb_rec.cpu().squeeze(0)

            score_vtm = logit_scale * (img_emb_replaced @ txt_emb.t())
            score_vtm_np = score_vtm.numpy()
            ret_vtm = eval_retrieval(score_vtm_np, img2txt, txt2img)

            for si, ti in enumerate(test_indices):
                rk = i2t_rank_for_image(score_vtm_np[ti], img2txt[ti])
                vtm_ranks.append(rk)

            avg_bpfp = np.mean(bpfp_list) if bpfp_list else 0
            avg_mse = np.mean(mse_list) if mse_list else 0
            avg_feat_cos = np.mean(feat_cos_list) if feat_cos_list else 0
            avg_emb_cos = np.mean(emb_cos_list) if emb_cos_list else 0

            all_results[(layer, qp)] = {
                'bpfp': avg_bpfp, 'feat_mse': avg_mse,
                'feat_cos': avg_feat_cos, 'emb_cos': avg_emb_cos,
                'ret': ret_vtm, 'ranks': vtm_ranks,
            }

            rank_str = "  ".join(
                f"[{sample_names[si]}] {anchor_ranks[si]}→{vtm_ranks[si]}"
                for si in range(N_SAMPLES))
            print(f"    QP={qp:>2}: BPFP={avg_bpfp:.4f}  "
                  f"FeatCos={avg_feat_cos:.4f}  EmbCos={avg_emb_cos:.4f}  "
                  f"i2t_R@1={ret_vtm['i2t_R@1']:.2f}%  I2T rank: {rank_str}")

    del model, vision_model
    torch.cuda.empty_cache()

    shutil.rmtree(tmp_dir, ignore_errors=True)

    # ================================================================
    #  Phase 4: 汇总报告
    # ================================================================
    print("\n" + "=" * 100)
    print("[4/4] 验证汇总报告")
    print("=" * 100)

    # 表 1: VTM 编码指标
    print(f"\n{'Layer':<8} {'QP':>4} {'BPFP':>8} {'FeatMSE':>10} "
          f"{'FeatCos':>8} {'EmbCos':>8}  "
          f"{'i2t_R@1':>8} {'i2t_R@5':>8} {'i2t_R@10':>9}  "
          f"{'t2i_R@1':>8} {'t2i_R@5':>8} {'t2i_R@10':>9}")
    print("-" * 110)

    for layer in LAYERS:
        # anchor row
        anc = all_results[(layer, "anchor")]['ret']
        print(f"{layer:<8} {'anc':>4} {'---':>8} {'---':>10} "
              f"{'1.0000':>8} {'1.0000':>8}  "
              f"{anc['i2t_R@1']:>7.2f}% {anc['i2t_R@5']:>7.2f}% {anc['i2t_R@10']:>8.2f}%  "
              f"{anc['t2i_R@1']:>7.2f}% {anc['t2i_R@5']:>7.2f}% {anc['t2i_R@10']:>8.2f}%")

        qps = QPS_MAP[layer]
        for qp in qps:
            r = all_results[(layer, qp)]
            rv = r['ret']
            print(f"{'':<8} {qp:>4} {r['bpfp']:>8.4f} {r['feat_mse']:>10.4f} "
                  f"{r['feat_cos']:>8.4f} {r['emb_cos']:>8.4f}  "
                  f"{rv['i2t_R@1']:>7.2f}% {rv['i2t_R@5']:>7.2f}% {rv['i2t_R@10']:>8.2f}%  "
                  f"{rv['t2i_R@1']:>7.2f}% {rv['t2i_R@5']:>7.2f}% {rv['t2i_R@10']:>8.2f}%")
        print()

    # 表 2: 每张测试图 I2T rank 变化
    print("\n─── 测试样本 I2T Rank 变化 ───")
    print(f"{'Layer':<8} {'QP':>4}  ", end="")
    for name in sample_names:
        print(f"  {name[-6:]}", end="")
    print()
    print("-" * (16 + 8 * N_SAMPLES))

    for layer in LAYERS:
        anc_ranks = all_results[(layer, "anchor")]['ranks']
        print(f"{layer:<8} {'anc':>4}  ", end="")
        for rk in anc_ranks:
            print(f"  {rk:>6}", end="")
        print()

        for qp in QPS_MAP[layer]:
            vtm_ranks = all_results[(layer, qp)]['ranks']
            print(f"{'':<8} {qp:>4}  ", end="")
            for rk in vtm_ranks:
                print(f"  {rk:>6}", end="")
            print()
        print()

    # 判定
    print("─── 验证结论 ───")

    print(f"\n  1. 各层 BPFP 覆盖情况 (目标: 0.05 ~ min(0.75, 无损)):")
    for layer in LAYERS:
        qps = QPS_MAP[layer]
        bpfps = [all_results[(layer, q)]['bpfp'] for q in qps]
        emb_coss = [all_results[(layer, q)]['emb_cos'] for q in qps]
        lossless_bpfp = None
        for bp, ec in zip(bpfps, emb_coss):
            if ec > 0.9999:
                lossless_bpfp = bp
                break
        bmin, bmax = min(bpfps), max(bpfps)
        ll_str = f"lossless@{lossless_bpfp:.4f}" if lossless_bpfp else "无QP达无损"
        ok = bmin <= 0.06 and (bmax >= 0.70 or lossless_bpfp is not None)
        sym = "OK" if ok else "ADJUST"
        print(f"     {layer}: [{bmin:.4f}, {bmax:.4f}]  {ll_str}  [{sym}]")

    print()
    print("  2. 流水线正确性（最低QP → EmbCosSim）:")
    for layer in LAYERS:
        qp_min = min(QPS_MAP[layer])
        r = all_results[(layer, qp_min)]
        ok = r['emb_cos'] > 0.999
        sym = "PASS" if ok else "FAIL"
        print(f"     {layer} QP={qp_min}: EmbCosSim={r['emb_cos']:.6f}  [{sym}]")

    print()
    all_emb_cos = [all_results[(l, q)]['emb_cos']
                   for l in LAYERS for q in QPS_MAP[l]]
    cos_min = min(all_emb_cos)
    print(f"  3. 最低 EmbCosSim: {cos_min:.6f}", end="")
    if cos_min > 0.99:
        print("  (高保真)")
    elif cos_min > 0.9:
        print("  (合理)")
    elif cos_min > 0.8:
        print("  (有一定失真)")
    else:
        print("  (失真较大)")

    print("\n" + "=" * 100)


if __name__ == "__main__":
    main()

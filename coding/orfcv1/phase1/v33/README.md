# V33 两阶段分配求解

训练、探针、搜索与交付验证。**本模块不自动跑完整 100 epoch ORFC 对比**；交付后用 `verify` 做同图 Tail MSE。

## Checkpoint 协议 `v33_codec_v1`

训练与工具共用（`config.CHECKPOINT_FORMAT` / `checkpoint.py`）：

```python
{
  "format": "v33_codec_v1",
  "geometry": {
    "groups": 32,
    "mode_bits": [1, 2, 3],
    "dim": 32,
    "parameterization": "direct" | "orfc_cayley" | "anchored_cayley"
  },
  "state_dict": <V33Codec.state_dict()>,
  "meta": {
    "step": int, "phase": 1 | 2,
    "allocation": list[int] | None,  # phase-2 固定分配
    "u0_frozen": bool, "anchor": str, "run_id": str, ...
  },
  "optimizer_state": optional  # 仅 resume；探针/搜索忽略
}
```

旁路文件：有 allocation 时写 `allocation.npy` + `*.json` 摘要。

API：`save_checkpoint` / `load_checkpoint` / `load_u0_rotation` / `load_allocation`。

## 推荐管线顺序

```bash
# cwd: featcodec/ORFC/coding/orfcv1
python -m phase1.v33.train --phase 1 --anchor R64 --run-id <RUN>
python -m phase1.v33.noise_floor --checkpoint …/checkpoint.pt --out …/noise.json
python -m phase1.v33.switch_probe --checkpoints <a> <b> <c> \
  --noise-floor-json …/noise.json --out …/probe.json
python -m phase1.v33.search --checkpoint <switch.pt> --out …/search.json
# → allocation.npy
python -m phase1.v33.train --phase 2 --anchor R64 --run-id <RUN>_p2 \
  --source …/phase1/checkpoint.pt --allocation …/allocation.npy
python -m phase1.v33.verify \
  --checkpoint …/phase2/checkpoint.pt --allocation …/allocation.npy \
  --out …/verify.json
```

打印同一说明：`python -m phase1.v33.verify --print-pipeline`。

## 模块与关键 API

| 模块 | 关键入口 |
|------|----------|
| `train` | `--phase {1,2}`；phase2 用 `--source` + `--allocation` |
| `noise_floor` | `calibrate`, `estimate_sigma_noise`, `run_calibration` |
| `switch_probe` | `group_subspace_overlap`, `record_pair`, `suggest_switch` |
| `search` | `run_search`, `one_opt_certificate`, `two_group_transfers` |
| `verify` | 1-opt、同图 vs ORFC、top-k 排序钩子、名义码率 |
| `valset` | `fixed_val_rows`, `load_val_resident`（对齐 v12 `N_VAL=500`） |
| `ranking` | `sparse_kendall_tau` |

## Verify

```bash
# 合成 smoke（无特征缓存）
python -m phase1.v33.verify --smoke --out /tmp/verify.json

# 交付：穷尽邻域 1-opt + train_val 同图 Tail MSE vs ORFC
python -m phase1.v33.verify \
  --checkpoint path/to/phase2/checkpoint.pt \
  --allocation path/to/allocation.npy \
  --orfc-ref blk20_K4_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42 \
  --out verify.json

# 默认 ORFC：coding/orfc/checkpoints/dinov2_vitl14/ 下与 R64/R96 匹配的 stem
# 可用 --orfc-ref STEM|.pt|.npz，或 --orfc-dir DIR；缺参考时用 --skip-orfc

# top-k 排序一致性（完整重训分数由外部提供，本入口不跑重训）
python -m phase1.v33.verify ... \
  --search-ranking search_topk.json \
  --retrain-scores retrain_topk.json
# JSON: [s0,s1,…] 或 {"ids":[…],"scores":[…]}  （越低越好，同序）
```

名义码率必须精确等于 `anchor.rate`。rANS / 下游未在 v33 内实现：见报告里 `rans_downstream` 的 TODO 与调用点（`v12.eval_downstream`、CoFAI rANS 示例）；可用 `--rans-cmd` / `--downstream-cmd` 只记录不执行。

## 探针 / 搜索 smoke

```bash
python -m phase1.v33.noise_floor --smoke --out /tmp/noise.json
python -m phase1.v33.switch_probe --smoke --out /tmp/probe.json
python -m phase1.v33.search --smoke --out /tmp/search.json
python -m phase1.v33.test_tools
```

## 注意

- 求值路径一次一个分配（`distortion.evaluate` 流式），不批量折叠。
- 切换曲线须**跑过峰值**；稳定 ≠ 正确性（见 `switch_probe` 的 `note`）。
- 代价表只用于提议 top-50；接受完全由真实 Tail MSE 决定。

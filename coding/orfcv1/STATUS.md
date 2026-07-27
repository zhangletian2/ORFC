# ORFC-v1 远端仓库状态

审计日期：2026-07-27  
远端目录：`/data4/workspace/zlt/featcodec/ORFC/coding/orfcv1`

## 当前入口

- 核心实现：`codec_v1.py`、`elastic.py`、`train_v1.py`
- v1.1 训练与评估：`run_v1_1.py`、`run_exp_v1_1.sh`、
  `run_exp_v1_1_parallel.sh`、`select_v1_1.py`
- 下游评估：`eval_v1_1_all.py`、`eval_seg_v1.py`
- 契约测试：`test_v1_contracts.py`
- 历史 v1 正式流程：`run_v1.py`、`run_exp_v1.sh` 及其选择、smoke、
  benchmark、guard 脚本

## 保留的结果

- `results/dinov2_vitl14/formal_k8_20260726T175547Z/`
  - ORFC-v1 正式 K8 交付、审计和下游结果；
- `results/dinov2_vitl14/v1_1_20260726T011657Z/`
  - v1.1 K8 validation 候选、选择结果和 U-only 对照；
- `results/dinov2_vitl14/eval_compare_20260726T041812/`
  - 已完成的分类和分割比较；
- `results/dinov2_vitl14/chunkbench_frozen_20260726/`
  - v1 探针分块性能基准。

正式 checkpoint 和 v1.1 checkpoint 均保留。seed 42 的正式
train/validation/test feature 与 teacher cache 保留，后续 K8/K256 共用。

## 已删除的历史

- 三轮 smoke 的结果、日志和 checkpoint；
- 正式运行前的根目录 warmup、smoke 和 Phase A 试跑；
- 第一次未产出结果的 `eval_compare_20260726T041733`；
- v1.1 并行脚本中止时的冗余 driver 日志；
- seed 43/44 及 smoke 使用的可再生 feature/teacher 大缓存；
- Python `__pycache__`。

仓库占用由约 34 GB 降至约 12 GB。删除的大缓存不能直接恢复，但可以根据
保留的 split manifest 和原始特征重新生成。

## 当前验证状态

- Python 与 shell 静态检查通过；
- `test_v1_contracts.py` 在 CUDA 上全部通过；
- 正式 K8、v1.1 选择结果及下一阶段脚本依赖文件均存在；
- 当前没有运行中的 ORFC-v1 训练或评估进程。

## 重要限制

`orfcv1` 当前仍是上层 Git 仓库中的未跟踪目录。当前源码修改时间晚于已有
v1.1 结果，无法仅靠 Git commit 将现有结果与当前源码一一对应。下一次正式
实验前应将当前源码纳入版本控制，或者在每个 run 目录保存一份只读源码副本。


# ORFC: Optimized Rotation-based Feature Codec

基于可微乘积量化 (Soft Product Quantization) 的视觉特征压缩框架，支持 DINOv2 和 CLIP 骨干网络。

## 目录结构

```
ORFC/
├── backbone/                   # 骨干网络源码
│   ├── clip/                   # OpenAI CLIP ViT-L/14
│   └── dinov2/                 # Meta DINOv2 ViT-L/14 & ViT-G/14
├── coding/
│   ├── orfc/                   # 核心方法: Soft PQ 编解码器
│   │   └── checkpoints/        # 预训练 ORFC codec 权重 (需用户下载, ~1.4 GB)
│   ├── chen2019/               # 基线: HM-16.21 (HEVC)
│   ├── CompressAI/             # 基线: 学习型图像压缩 (Hyperprior)
│   └── vtm_baseline/           # 基线: VTM (VVC)
├── tools/                      # 特征提取脚本
├── utils/                      # 工具脚本、标签文件、评估配置
├── data/                       # 数据集 (需用户准备)
├── features/                   # 预提取特征 (需用户生成)
├── pretrained/                 # 骨干网络预训练权重 (需用户下载)
├── environment.yml             # Conda 环境定义
└── README.md
```

## 快速开始

### 1. 环境配置

```bash
# 创建 conda 环境 (首次安装约 40 分钟，主要耗时在 mmcv-full CUDA 扩展编译)
conda env create -f environment.yml
conda activate featcodec2

# 安装 CLIP (develop 模式)
cd backbone/clip && pip install -e . && cd ../..

# 安装 CompressAI (develop 模式，需编译 C++ 扩展，约 9 分钟)
cd coding/CompressAI && pip install -e . && cd ../..
```

### 2. 预训练权重

#### DINOv2 权重

下载 DINOv2 权重并设置 torch.hub 兼容目录结构:

```bash
mkdir -p pretrained/hub/checkpoints && cd pretrained

# DINOv2 ViT-L/14 (必需，~1.2 GB)
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_linear_head.pth

# DINOv2 ViT-G/14 (可选，ViT-G 实验需要，~4.3 GB)
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_linear_head.pth

# 创建 hub/checkpoints 链接 (torch.hub 加载兼容)
cd hub/checkpoints && ln -sf ../../*.pth . && cd ../../..
```

> DINOv2 分割头权重 (`*_voc2012_linear_head.pth`) 请参考 [DINOv2 官方仓库](https://github.com/facebookresearch/dinov2) 获取，同样放入 `pretrained/` 和 `pretrained/hub/checkpoints/`。

DINOv2 完整权重列表:

| 文件 | 大小 | 用途 |
|------|------|------|
| `dinov2_vitl14_pretrain.pth` | 1.2 GB | ViT-L backbone |
| `dinov2_vitl14_linear_head.pth` | 7.9 MB | ViT-L 分类头 |
| `dinov2_vitl14_voc2012_linear_head.pth` | 295 KB | ViT-L 分割头 |
| `dinov2_vitg14_pretrain.pth` | 4.3 GB | ViT-G backbone |
| `dinov2_vitg14_linear_head.pth` | 12 MB | ViT-G 分类头 |
| `dinov2_vitg14_voc2012_linear_head.pth` | 437 KB | ViT-G 分割头 |

#### CLIP 权重

CLIP ViT-L/14 权重在首次调用 `clip.load("ViT-L/14")` 时自动下载到 `~/.cache/clip/`（约 890 MB），无需手动准备。也可以预先下载:

```bash
mkdir -p ~/.cache/clip && cd ~/.cache/clip
wget https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
cd -
```

### 3. ORFC Codec 预训练权重 (可选)

论文所有 Rate-Distortion 曲线与消融实验的 Soft PQ codec checkpoints 已整理发布。如果你只想**复现评估 / 画图 / 用已训好的 codec 压缩特征**，无需重训，直接下载即可。

#### 下载地址

- **阿里云盘**: `https://www.alipan.com/s/<TBD>`（分享码 `TBD`，共 **1.4 GB**）

下载后按目录结构放入 `coding/orfc/checkpoints/`:

```
coding/orfc/checkpoints/
├── clip_vitl14/          #   4 个 .pt, ~41 MB  (Exp 1 sensitivity: blk{05,10,15,20}, K=16)
├── dinov2_vitl14/        #  73 个 .pt, ~763 MB (论文主 RD 曲线 + Exp 1/3/4/5/6 消融)
└── dinov2_vitg14/        #  27 个 .pt, ~620 MB (ViT-G/14 主 RD 曲线 + 多 seed)
```

> 阿里云盘不支持命令行直接下载，请通过浏览器打开链接下载压缩包后手动解压到上述路径，或逐目录同步。

#### 验证

```bash
cd coding/orfc
ls checkpoints/dinov2_vitl14/ | wc -l    # 应为 73
ls checkpoints/dinov2_vitg14/ | wc -l    # 应为 27
ls checkpoints/clip_vitl14/   | wc -l    # 应为  4
```

#### 命名规则

```
{layer}_K{K}_emb{d}_bt{bt_dim}_{ws|km}[{_mse}][{_lmbda*}][{_fzR}][{_rot*}][{_tau*}][{_te*}][{_ts*}]_lr{lr}_ep{ep}_n{n}_s{seed}.pt
```

| 字段 | 含义 |
|------|------|
| `layer` | 压缩的 backbone 层, 如 `blk20` 代表 DINOv2 ViT-L 第 20 个 block 输出 |
| `K`, `emb` | PQ 的码本大小 K 和子向量维度 d |
| `bt{bt_dim}` | 输入特征维度 (ViT-L=1024, ViT-G=1536) |
| `ws` / `km` | OPQ warm-start / 纯 k-means 初始化 |
| `_mse` | 仅 MSE 损失 (消融) |
| `_lmbda*` | 速率正则权重 λ (无此标则 λ=0) |
| `_fzR` | 冻结旋转矩阵 R (消融) |
| `_rot{pca,randomorth,identity}` | 替换 OPQ 为其它初始化 (Exp 5) |
| `_tau*`, `_te*`, `_ts*` | 温度退火起点 / 终点 / schedule (Exp 4) |
| `_lr*`, `_ep*`, `_s*` | 学习率 / 训练轮数 / 随机种子 |

每个 `.pt` 对应的评估结果（Acc, mIoU, MSE, rANS BPT）都保存在 `results/soft_pq/{backbone}/<同名>.json` 中，训练 / 配置信息写在 json 的 `config` 字段里。

#### Checkpoint 清单

论文主要的最优 RD 点、Exp 1-6 的消融实验均覆盖，共 **104 个 checkpoint**；**最优点清单**见 `coding/orfc/best config.csv`，**批量训练清单**见 `coding/orfc/experiment_manifest.json`。

### 4. 数据集

**Pascal VOC 2012** (分割评估):

```bash
mkdir -p data && cd data
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_11-May-2012.tar
cd ..
```

**ImageNet ILSVRC2012 Validation Set** (特征提取 / 分类评估):

从 [ImageNet 官方](https://image-net.org/) 下载验证集，解压后应为 `<imagenet_root>/<wnid>/ILSVRC2012_val_*.JPEG` 格式。

### 5. 特征提取

本框架使用两个独立的特征提取脚本，分别对应不同数据集格式:

#### ImageNet 分类特征 (双列列表: `<wnid> <basename>`)

```bash
export TORCH_HOME=$(pwd)/pretrained

# DINOv2 ViT-L/14: 训练集 (5000 张, 4 层)
python tools/dinov2_feat_pipeline_simple.py extract \
    --model vitl14 \
    --weights_root pretrained \
    --root /path/to/imagenet/val \
    --out_root features/train/dinov2_vitl14 \
    --list utils/imagenet_selected_pathname5000.txt \
    --blocks 5,10,15,20

# CLIP ViT-L/14: 训练集 (5000 张, 4 层)
python tools/clip_feat_pipeline_simple.py extract \
    --root /path/to/imagenet/val \
    --out_root features/train/clip_vitl14 \
    --list utils/imagenet_selected_pathname5000.txt \
    --blocks 5,10,15,20
```

#### VOC 分割特征 (单列列表: `<basename>`，滑窗模式)

```bash
export TORCH_HOME=$(pwd)/pretrained

# DINOv2 ViT-L/14: VOC2012 评估集 (100 张, 滑窗)
python tools/dinov2_seg_pipeline.py extract \
    --model vitl14 \
    --out_root features/voc2012_100/dinov2_vitl14 \
    --blocks 5,10,15,20 \
    --image_list utils/voc2012_val_100.txt

# DINOv2 ViT-L/14: VOC2012 训练集 (5000 张, 滑窗)
python tools/dinov2_seg_pipeline.py extract \
    --model vitl14 \
    --out_root features/voc2012_5000/dinov2_vitl14 \
    --blocks 5,10,15,20 \
    --image_list utils/voc2012_all_5000.txt
```

#### 特征目录结构

```
features/
├── train/<backbone>/blk{05,10,15,20}/*.npy        # ImageNet 训练用
├── val/<backbone>/blk{05,10,15,20}/*.npy           # ImageNet 分类评估用
├── voc2012_100/<backbone>/blk{05,10,15,20}/*.npy   # VOC 分割评估用 (100 张)
└── voc2012_5000/<backbone>/blk{05,10,15,20}/*.npy  # VOC 分割训练用 (5000 张)
```

每个 `.npy` 文件形状:
- ImageNet 特征: `[N_tokens, D]`（如 ViT-L: `[257, 1024]`）
- VOC 分割特征: `[N_slides, 1+N, D]`（滑窗模式，含 CLS token）

## 使用方法

### ORFC: Soft PQ 编解码器 (核心方法)

#### 用预训练 codec 直接评估（无需训练）

下载好 `coding/orfc/checkpoints/` 后即可复用已训好的 codec 做评估 / 分析：

```bash
cd coding/orfc

# 仅评估（跳过训练），复用已下载的 checkpoint
#   --ckpt_path 里的超参 (K, emb, bt, lmbda, tau, lr, ep, seed) 必须与文件名一致
python run_soft_pq.py \
    --backbone dinov2_vitl14 --layer blk20 \
    --K 16 --embedding_dim 32 --bottleneck_dim 1024 --warm_start_opq \
    --lmbda 0.5 --tau_start 0.5 --tau_end 0.005 \
    --lr 3e-4 --epochs 100 --max_train_images 5000 --eval_seg \
    --eval_only \
    --ckpt_path checkpoints/dinov2_vitl14/blk20_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt

# 灵敏度 / 温度 / 初始化 / 多 seed 分析脚本也都基于已有 checkpoint：
python compute_sensitivity.py --gpu 0 \
    --codec_path checkpoints/dinov2_vitl14/blk20_K16_emb32_bt1024_ws_lmbda0.5_tau0.5_lr0.0003_ep100_n5000_s42.pt
python compute_multi_sensitivity.py --gpu 0
python plot_exp4_temperature.py   # Exp 4 温度退火
python plot_exp5_init.py          # Exp 5 初始化
python plot_exp6_seed.py          # Exp 6 多 seed
```

#### 从零训练

```bash
cd coding/orfc

# 单次训练 + 评估 (DINOv2 ViT-L/14, block 20)
python run_soft_pq.py \
    --backbone dinov2_vitl14 \
    --layer blk20 \
    --K 16 --embedding_dim 32 \
    --bottleneck_dim 1024 --warm_start_opq \
    --lmbda 0.5 --tau_start 0.5 --tau_end 0.005 \
    --epochs 100 --max_train_images 5000 \
    --eval_seg

# CLIP 实验 (需指定 classnames)
python run_soft_pq.py \
    --backbone clip_vitl14 \
    --layer blk20 \
    --K 16 --embedding_dim 32 \
    --bottleneck_dim 768 --warm_start_opq \
    --lmbda 0.5 --epochs 100

# 多层 + 多配置批量训练 (ViT-G/14, 6 GPU)
bash run_exp_vitg14.sh

# 批量训练 / 评估
python run_batch_train.py --backbone dinov2_vitl14
python run_batch_eval.py --backbone dinov2_vitl14

# 消融实验
bash run_ablation.sh
```

### CompressAI: Hyperprior 基线

```bash
cd coding/CompressAI

# 分割任务 (DINOv2 ViT-L/14)
bash run_hyperprior_seg.sh

# 分类任务
bash run_hyperprior_cls.sh

# CLIP 分类任务
bash run_hyperprior_cls_clip.sh
```

### VTM: VVC 编解码基线

```bash
cd coding/vtm_baseline

# 分割特征编解码 (多 QP)
bash run_vtm_seg.sh

# 分类特征编解码
bash run_vtm.sh
```

### 评估与可视化

```bash
cd coding/orfc

# 分类精度 vs 码率曲线
python eval_cls_rate.py

# 分割 mIoU vs 码率曲线
python eval_voc_rate.py

# 编解码延时测试
python eval_timing.py

# 绘制论文图表
python plot_exp1_sensitivity.py
python plot_exp2_correlation.py
python plot_exp3_gkd.py
```

## 核心代码说明

| 文件 | 功能 |
|------|------|
| `coding/orfc/soft_pq.py` | Soft PQ 编解码器核心实现（可微温度退火、正交旋转） |
| `coding/orfc/opq.py` | OPQ 旋转矩阵学习、特征归一化 |
| `coding/orfc/run_soft_pq.py` | 训练和评估主入口 |
| `coding/orfc/backbone/wrapper.py` | DINOv2 / CLIP 推理封装（分类 + 分割） |
| `coding/orfc/run_multilayer_calibrator.py` | 多层特征校准器 |
| `coding/orfc/entropy_coding.py` | ANS 熵编码 |
| `coding/orfc/simple_fcvq.py` | 基础 VQ 编码器 |
| `coding/orfc/metric_estimator.py` | 下游任务指标估计 |
| `tools/dinov2_feat_pipeline_simple.py` | DINOv2 分类特征提取 (ImageNet，extract + replay) |
| `tools/clip_feat_pipeline_simple.py` | CLIP 特征提取 (ImageNet，extract + replay) |
| `tools/dinov2_seg_pipeline.py` | DINOv2 分割特征提取 (VOC，滑窗 extract + replay) |
| `utils/classnames.txt` | ImageNet 1000 类名 (CLIP zero-shot 用) |
| `utils/cal_bd_rate.py` | BD-Rate 计算工具 |

# ORFC: Optimized Rotation-based Feature Codec

基于可微乘积量化 (Soft Product Quantization) 的视觉特征压缩框架，支持 DINOv2 和 CLIP 骨干网络。

## 目录结构

```
ORFC/
├── backbone/          # 骨干网络源码
│   ├── clip/          # OpenAI CLIP ViT-L/14
│   └── dinov2/        # Meta DINOv2 ViT-L/14 & ViT-G/14
├── coding/
│   ├── orfc/          # 核心方法: Soft PQ 编解码器
│   ├── chen2019/      # 基线: HM-16.21 (HEVC)
│   ├── CompressAI/    # 基线: 学习型图像压缩 (Hyperprior)
│   └── vtm_baseline/  # 基线: VTM (VVC)
├── tools/             # 特征提取脚本
├── utils/             # 工具脚本、标签文件、评估配置
├── data/              # 数据集 (需用户准备)
├── features/          # 预提取特征 (需用户生成)
├── pretrained/        # 预训练权重 (需用户下载)
├── environment.yml    # Conda 环境定义
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

# 安装 CompressAI (develop 模式，约 9 分钟)
cd coding/CompressAI && pip install -e . && cd ../..
```

### 2. 预训练权重

下载权重并设置目录结构:

```bash
mkdir -p pretrained/hub/checkpoints && cd pretrained

# DINOv2 ViT-L/14 (必需，~1.2 GB)
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_linear_head.pth

# DINOv2 ViT-G/14 (可选，ViT-G 实验需要，~4.3 GB)
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_pretrain.pth
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_linear_head.pth

# 创建 hub/checkpoints 链接 (torch.hub 兼容)
cd hub/checkpoints && ln -sf ../../*.pth . && cd ../../..
```

> DINOv2 分割头权重 (`*_voc2012_linear_head.pth`) 请参考 [DINOv2 官方仓库](https://github.com/facebookresearch/dinov2) 获取，同样放入 `pretrained/` 和 `pretrained/hub/checkpoints/`。

完整权重列表:

| 文件 | 大小 | 用途 |
|------|------|------|
| `dinov2_vitl14_pretrain.pth` | 1.2 GB | ViT-L backbone |
| `dinov2_vitl14_linear_head.pth` | 7.9 MB | ViT-L 分类头 |
| `dinov2_vitl14_voc2012_linear_head.pth` | 295 KB | ViT-L 分割头 |
| `dinov2_vitg14_pretrain.pth` | 4.3 GB | ViT-G backbone |
| `dinov2_vitg14_linear_head.pth` | 12 MB | ViT-G 分类头 |
| `dinov2_vitg14_voc2012_linear_head.pth` | 437 KB | ViT-G 分割头 |

### 3. 数据集

**Pascal VOC 2012** (分割评估):

```bash
mkdir -p data && cd data
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar xf VOCtrainval_11-May-2012.tar
cd ..
```

**ImageNet ILSVRC2012 Validation Set** (特征提取 / 分类评估):

从 [ImageNet 官方](https://image-net.org/) 下载验证集，解压后应为 `<imagenet_root>/<wnid>/ILSVRC2012_val_*.JPEG` 格式。

### 4. 特征提取

从原始图像提取 ViT 中间层特征:

```bash
# DINOv2 ViT-L/14: 训练集 (5000 张, 4 个 block)
python tools/dinov2_feat_pipeline_simple.py extract \
    --model vitl14 \
    --weights_root pretrained \
    --root /path/to/imagenet/val \
    --out_root features/train/dinov2_vitl14 \
    --list utils/imagenet_selected_pathname5000.txt \
    --blocks 5 10 15 20

# DINOv2 ViT-L/14: VOC2012 分割评估集 (100 张)
python tools/dinov2_feat_pipeline_simple.py extract \
    --model vitl14 \
    --weights_root pretrained \
    --root data/VOCdevkit/VOC2012/JPEGImages \
    --out_root features/voc2012_100/dinov2_vitl14 \
    --list utils/voc2012_val_100.txt \
    --blocks 5 10 15 20

# CLIP ViT-L/14: 训练集
python tools/clip_feat_pipeline_simple.py extract \
    --root /path/to/imagenet/val \
    --out_root features/train/clip_vitl14 \
    --list utils/imagenet_selected_pathname5000.txt \
    --blocks 5 10 15 20
```

特征目录结构:
```
features/
├── train/<backbone>/blk{05,10,15,20}/*.npy   # 训练用
├── val/<backbone>/blk{05,10,15,20}/*.npy      # 分类评估用
└── voc2012_100/<backbone>/blk{05,10,15,20}/*.npy  # 分割评估用
```

每个 `.npy` 文件形状为 `[N_tokens, D]`（如 ViT-L: `[257, 1024]`，ViT-G: `[257, 1536]`）。

## 使用方法

### ORFC: Soft PQ 编解码器 (核心方法)

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
| `tools/dinov2_feat_pipeline_simple.py` | DINOv2 特征提取 (extract + replay) |
| `tools/clip_feat_pipeline_simple.py` | CLIP 特征提取 (extract + replay) |
| `tools/dinov2_seg_pipeline.py` | DINOv2 分割特征评估管线 |
| `utils/classnames.txt` | ImageNet 1000 类名 (CLIP zero-shot 用) |
| `utils/cal_bd_rate.py` | BD-Rate 计算工具 |

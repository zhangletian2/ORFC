import os
import sys
import math
import itertools
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_BACKBONE_DIR = os.path.dirname(os.path.abspath(__file__))
_ORFC_ROOT = os.path.dirname(_BACKBONE_DIR)
_PROJECT_ROOT = os.path.normpath(os.path.join(_ORFC_ROOT, "..", ".."))
DINOV2_SOURCE_DIR = os.path.join(_PROJECT_ROOT, "backbone", "dinov2")
if os.path.exists(DINOV2_SOURCE_DIR):
    sys.path.append(DINOV2_SOURCE_DIR)
    try:
        from dinov2.hub.classifiers import dinov2_vitl14_lc, dinov2_vitg14_lc
    except Exception as e:
                print(f"Failed to import DINOv2: {e}")
else:
    print(f"DINOv2 source not found at {DINOV2_SOURCE_DIR}")

UTILS_DIR = os.path.join(_PROJECT_ROOT, "utils")

_DINOV2_REGISTRY = {
    "dinov2_vitl14": {
        "lc_fn": dinov2_vitl14_lc,
        "pretrain": "dinov2_vitl14_pretrain.pth",
        "linear_head": "dinov2_vitl14_linear_head.pth",
        "seg_head": "dinov2_vitl14_voc2012_linear_head.pth",
        "embed_dim": 1024,
        "config": os.path.join(UTILS_DIR, "dinov2_vitl14_voc2012_linear_config.py"),
        "vit_fn": "vit_large",
        "vit_kwargs": dict(patch_size=14, img_size=518, init_values=1.0, block_chunks=0),
    },
    "dinov2_vitg14": {
        "lc_fn": dinov2_vitg14_lc,
        "pretrain": "dinov2_vitg14_pretrain.pth",
        "linear_head": "dinov2_vitg14_linear_head.pth",
        "seg_head": "dinov2_vitg14_voc2012_linear_head.pth",
        "embed_dim": 1536,
        "config": os.path.join(UTILS_DIR, "dinov2_vitg14_voc2012_linear_config.py"),
        "vit_fn": "vit_giant2",
        "vit_kwargs": dict(patch_size=14, img_size=518, init_values=1.0,
                           block_chunks=0, ffn_layer="swiglufused"),
    },
}

try:
    import clip as _clip_module
    _HAS_CLIP = True
except ImportError:
    _HAS_CLIP = False


# ========================= 分割头 =========================

class SegmentationHead(nn.Module):
    """
    BN + 1x1 Conv 分割头（与官方VOC2012 linear head一致）
    使用 SyncBatchNorm 以匹配官方权重
    """
    def __init__(self, in_channels=1024, num_classes=21):
        super().__init__()
        self.bn = nn.SyncBatchNorm(in_channels)
        self.conv_seg = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # x: [B, C, H, W]
        x = self.bn(x)
        x = self.conv_seg(x)
        return x


def load_seg_head(weights_path, in_channels=1024, num_classes=21, device='cuda'):
    """加载分割头权重"""
    head = SegmentationHead(in_channels, num_classes)
    ckpt = torch.load(weights_path, map_location='cpu')
    if 'state_dict' in ckpt:
        ckpt = ckpt['state_dict']
    # 转换key名称: decode_head.xxx -> xxx
    head_state = {}
    for k, v in ckpt.items():
        if k.startswith('decode_head.'):
            new_k = k.replace('decode_head.', '')
            head_state[new_k] = v
    head.load_state_dict(head_state, strict=True)
    return head.to(device).eval()

class ClipWrapper:
    """
    CLIP ViT-L/14 wrapper with the same interface as Dinov2Wrapper.

    Classification: zero-shot via cosine similarity with text embeddings.
    Tail blocks: visual.transformer.resblocks  (input/output: [L, B, D])
    """

    def __init__(self, classnames_path, device="cuda",
                 template="a photo of a {}"):
        assert _HAS_CLIP, "pip install git+https://github.com/openai/CLIP.git"
        model, _ = _clip_module.load("ViT-L/14", device=device)
        model.eval().float()
        for p in model.parameters():
            p.requires_grad_(False)

        self.model = model
        self.visual = model.visual
        self.device = device
        self.weights_root = None
        self.head = None

        self._resblocks = list(self.visual.transformer.resblocks)
        self._ln_post = self.visual.ln_post
        self._proj = self.visual.proj

        self.text_emb = self._build_text_emb(classnames_path, template)

    # --- mimic Dinov2Wrapper.backbone.{blocks, norm} ---

    class _BackboneView:
        """Thin proxy so that wrapper.backbone.blocks / .norm work."""
        def __init__(self, resblocks, ln_post):
            self.blocks = resblocks
            self.norm = ln_post
        def to(self, device):
            for b in self.blocks:
                b.to(device)
            self.norm.to(device)
            return self
        def cpu(self):
            return self.to('cpu')
        def parameters(self):
            for b in self.blocks:
                yield from b.parameters()
            yield from self.norm.parameters()

    @property
    def backbone(self):
        return self._BackboneView(self._resblocks, self._ln_post)

    # --- text embeddings ---

    @torch.no_grad()
    def _build_text_emb(self, classnames_path, template):
        names = []
        with open(classnames_path, 'r') as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                names.append(" ".join(parts[1:]))
        prompts = [template.format(n) for n in names]
        tokens = _clip_module.tokenize(prompts).to(self.device)
        text_feat = self.model.encode_text(tokens).float()
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return text_feat

    # --- classification (same signature as Dinov2Wrapper) ---

    @torch.no_grad()
    def forward_from_tokens(self, tokens, start_block_idx):
        """tokens: [B, T, D] -> logits [B, C]  (zero-shot)"""
        x = tokens.permute(1, 0, 2).contiguous()          # [T, B, D]
        for i in range(start_block_idx + 1, len(self._resblocks)):
            x = self._resblocks[i](x)
        x = x.permute(1, 0, 2)                            # [B, T, D]
        cls = x[:, 0, :]                                   # [B, D]
        cls = self._ln_post(cls)
        if isinstance(self._proj, torch.Tensor):
            img_feat = cls @ self._proj                    # [B, D_out]
        else:
            img_feat = self._proj(cls)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        logit_scale = self.model.logit_scale.exp()
        return logit_scale * img_feat @ self.text_emb.t()


try:
    from transformers import AutoModel as _AutoModel, AutoProcessor as _AutoProcessor
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False


class Siglip2Wrapper:
    """
    SigLIP2 So400m-patch14-224 wrapper (same interface as ClipWrapper / Dinov2Wrapper).

    Key differences from CLIP:
      - No CLS token; 256 pure patch tokens, D=1152
      - MultiheadAttentionPoolingHead (MAP) instead of CLS→proj
      - logits = scale * cos + bias
      - encoder.layers output is (hidden_states, ...) tuple
    """

    def __init__(self, classnames_path, device="cuda",
                 model_id="google/siglip2-so400m-patch14-224",
                 template="This is a photo of {}."):
        assert _HAS_TRANSFORMERS, "pip install transformers>=4.49 sentencepiece"
        import os as _os
        hf_mirror = _os.environ.get("HF_ENDPOINT", "")
        model = _AutoModel.from_pretrained(model_id).eval().float().to(device)
        processor = _AutoProcessor.from_pretrained(model_id)
        for p in model.parameters():
            p.requires_grad_(False)

        self.model = model
        self.processor = processor
        self.device = device
        self.weights_root = None
        self.head = None

        vm = model.vision_model
        self._layers = list(vm.encoder.layers)
        self._post_ln = vm.post_layernorm
        self._map_head = vm.head if hasattr(vm, 'head') else None
        self._logit_scale = model.logit_scale
        self._logit_bias = model.logit_bias

        self.text_emb = self._build_text_emb(classnames_path, template)
        model.text_model.cpu()
        torch.cuda.empty_cache()

    class _BackboneView:
        """Proxy so that wrapper.backbone.blocks / .norm work."""
        def __init__(self, layers, post_ln):
            self.blocks = layers
            self.norm = post_ln
        def to(self, device):
            for b in self.blocks:
                b.to(device)
            self.norm.to(device)
            return self
        def cpu(self):
            return self.to('cpu')
        def parameters(self):
            for b in self.blocks:
                yield from b.parameters()
            yield from self.norm.parameters()

    @property
    def backbone(self):
        return self._BackboneView(self._layers, self._post_ln)

    @torch.no_grad()
    def _build_text_emb(self, classnames_path, template):
        names = []
        with open(classnames_path, 'r') as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                parts = ln.split()
                names.append(" ".join(parts[1:]))
        prompts = [template.format(n) for n in names]
        text_inputs = self.processor(
            text=prompts, padding="max_length", max_length=64,
            truncation=True, return_tensors="pt",
        ).to(self.device)
        text_out = self.model.get_text_features(**text_inputs)
        text_emb = text_out.pooler_output if hasattr(text_out, 'pooler_output') else text_out
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        return text_emb.float()

    @torch.no_grad()
    def forward_from_tokens(self, tokens, start_block_idx):
        """tokens: [B, N, D] (N=256 patches, no CLS) -> logits [B, C]"""
        x = tokens
        for i in range(start_block_idx + 1, len(self._layers)):
            layer_out = self._layers[i](x, attention_mask=None)
            x = layer_out[0] if isinstance(layer_out, tuple) else layer_out
        x = self._post_ln(x)
        if self._map_head is not None:
            pooled = self._map_head(x)
        else:
            pooled = x.mean(dim=1)
        pooled = pooled / pooled.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        logit_scale = self._logit_scale.exp()
        return logit_scale * pooled @ self.text_emb.t() + self._logit_bias


class Dinov2Wrapper:
    """
    Forward tokens from current ViT block output -> target block output.
    Model params are frozen; gradients flow through the blocks into rec_feat.

    Supports dinov2_vitl14 (24 blocks, dim=1024) and dinov2_vitg14 (40 blocks, dim=1536).
    """
    def __init__(self, head_layers, model_name="dinov2_vitl14",
                 weights_root=os.path.join(_PROJECT_ROOT, "pretrained"),
                 device="cuda"):
        reg = _DINOV2_REGISTRY[model_name]
        self.model_name = model_name
        self.embed_dim = reg["embed_dim"]

        back = os.path.join(weights_root, reg["pretrain"])
        head = os.path.join(weights_root, reg["linear_head"])
        lc_fn = reg["lc_fn"]
        clf = lc_fn(layers=head_layers, pretrained=True, weights=[back, head])
        clf.to(device).eval()
        for p in clf.parameters():
            p.requires_grad_(False)
        self.backbone = getattr(clf, "backbone", clf)
        self.head = getattr(clf, "linear_head", None)
        self.seg_head = None
        self.device = device
        self.weights_root = weights_root

    def load_segmentation_head(self, seg_head_path=None, num_classes=21):
        reg = _DINOV2_REGISTRY[self.model_name]
        if seg_head_path is None:
            seg_head_path = os.path.join(self.weights_root, reg["seg_head"])
        self.seg_head = load_seg_head(
            seg_head_path, in_channels=self.embed_dim,
            num_classes=num_classes, device=self.device)
        print(f"  分割头已加载: {seg_head_path}")

    @torch.no_grad()
    def forward_from_tokens(self, tokens, start_block_idx):
        """
        DINOv2-L/14: 从中间层 tokens (含 CLS，[B,N,D]) 继续前向，
        构造官方 LinearClassifierWrapper(layers=1) 的输入：
            linear_input = cat([x_norm_clstoken, mean(x_norm_patchtokens)], dim=1)
        返回 logits: [B, num_classes]
        """
        x = tokens
        for i in range(start_block_idx + 1, len(self.backbone.blocks)):
            x = self.backbone.blocks[i](x)
        x = self.backbone.norm(x)
        cls_token = x[:, 0]          # [B, D]
        patch_tokens = x[:, 1:]      # [B, P, D]
        mean_patch = patch_tokens.mean(dim=1)  # [B, D]
        linear_input = torch.cat([cls_token, mean_patch], dim=1)

        return self.head(linear_input)

    @torch.no_grad()
    def forward_from_tokens_seg(self, tokens, start_block_idx, feat_h, feat_w):
        """
        分割任务：从中间层 tokens 继续前向，输出分割 logits
        
        Args:
            tokens: [B, 1+N, D] 含 CLS token 的特征
            start_block_idx: 起始 block 索引
            feat_h: 特征图高度（patch 数）
            feat_w: 特征图宽度（patch 数）
        
        Returns:
            logits: [B, num_classes, feat_h, feat_w] 分割 logits
        """
        assert self.seg_head is not None, "请先调用 load_segmentation_head 加载分割头"
        
        x = tokens
        for i in range(start_block_idx + 1, len(self.backbone.blocks)):
            x = self.backbone.blocks[i](x)
        x = self.backbone.norm(x)
        
        # 去掉 CLS token，只保留 patch tokens
        patch_tokens = x[:, 1:]  # [B, N, D]
        
        # Reshape 为空间格式
        B, N, D = patch_tokens.shape
        assert N == feat_h * feat_w, f"patch 数不匹配: {N} != {feat_h}*{feat_w}"
        feat = patch_tokens.reshape(B, feat_h, feat_w, D)  # [B, H, W, D]
        feat = feat.permute(0, 3, 1, 2)  # [B, D, H, W]
        
        # 分割头推理
        logits = self.seg_head(feat)  # [B, num_classes, H, W]
        
        return logits


# ========================= mmseg 官方评估支持 =========================

class CenterPadding(nn.Module):
    """将输入 pad 到 patch_size 的整数倍（官方实现）"""
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(
            self._get_pad(m) for m in x.shape[:1:-1]))
        return F.pad(x, pads)


class SegmentationEvaluator:
    """
    分割评估器（slide inference，参考 dinov2_seg_pipeline.py）
    
    流程：
    1. 加载预提取的特征 [num_slides, 1+N, D]
    2. 对每个 slide 的特征做量化 + 校准
    3. 继续前向 + norm → 分割头 → 滑窗融合
    4. 与 GT 比较，计算 mIoU
    
    仅依赖 mmcv/mmseg 的轻量组件（Config、pipeline 预处理）。
    """
    
    # VOC2012 常量
    VOC_CLASSES = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    NUM_CLASSES = 21
    IGNORE_INDEX = 255
    CROP_SIZE = (512, 512)
    STRIDE = (341, 341)
    PATCH_SIZE = 14
    
    def __init__(self, codec_list, calibrator, layer_idx,
                 voc_root, weights_root, device='cuda',
                 norm_mode='per_token_ln', feat_dim=1024,
                 model_name='dinov2_vitl14'):
        self.codec_list = codec_list
        self.calibrator = calibrator
        self.layer_idx = layer_idx
        self.voc_root = voc_root
        self.weights_root = weights_root
        self.device = device
        self.norm_mode = norm_mode
        self.feat_dim = feat_dim
        self.model_name = model_name
    
    # ============== 归一化 ==============
    
    def _per_token_layernorm(self, x, eps=1e-5):
        mu = x.mean(axis=1, keepdims=True)
        var = ((x - mu) ** 2).mean(axis=1, keepdims=True)
        std = np.sqrt(var + eps)
        return ((x - mu) / std).astype(np.float32), mu.astype(np.float32), std.astype(np.float32)
    
    def _inv_per_token_layernorm(self, y_hat, mu, std):
        return (y_hat * std + mu).astype(np.float32)
    
    def _per_image_norm(self, x, eps=1e-5):
        mu_full = x.mean()
        var = ((x - mu_full) ** 2).mean()
        std = np.sqrt(var + eps)
        return ((x - mu_full) / std).astype(np.float32), \
               np.array([[mu_full]], dtype=np.float32), \
               np.array([[std]], dtype=np.float32)
    
    def _inv_per_image_norm(self, y_hat, mu, std):
        return (y_hat * std + mu).astype(np.float32)
    
    def normalize(self, x, mode='per_token_ln'):
        if mode == 'per_token_ln':
            return self._per_token_layernorm(x)
        elif mode == 'per_image':
            return self._per_image_norm(x)
        raise ValueError(f"Unknown norm_mode: {mode}")
    
    def inv_normalize(self, y_hat, mu, std, mode='per_token_ln'):
        if mode == 'per_token_ln':
            return self._inv_per_token_layernorm(y_hat, mu, std)
        elif mode == 'per_image':
            return self._inv_per_image_norm(y_hat, mu, std)
        raise ValueError(f"Unknown norm_mode: {mode}")
    
    # ============== 量化 ==============
    
    def quantize_tokens(self, tokens_np):
        """
        量化单个 slide 的 tokens: [1+N, D] → [1+N, D]
        """
        y, mu, std = self.normalize(tokens_np, mode=self.norm_mode)
        y_hat = np.zeros_like(y)
        for codec in self.codec_list:
            if codec is None:
                continue
            r = y - y_hat
            r_t = torch.from_numpy(r.T).float().to(self.device)
            with torch.no_grad():
                outputs = codec(r_t.unsqueeze(0))
                r_hat = outputs[0].squeeze(0).T.cpu().numpy()
            y_hat = y_hat + r_hat
        x_hat = self.inv_normalize(y_hat, mu, std, mode=self.norm_mode)
        x_hat_t = torch.from_numpy(x_hat).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            x_cal = self.calibrator(x_hat_t).squeeze(0)
        return x_cal  # [1+N, D] tensor on device
    
    # ============== Slide inference ==============
    
    @staticmethod
    def get_slide_crops(h_img, w_img, crop_size, stride):
        """计算滑窗裁剪区域，返回 list of (y1, x1, y2, x2)"""
        h_crop, w_crop = crop_size
        h_stride, w_stride = stride
        crops = []
        for h_idx in range(0, max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1):
            for w_idx in range(0, max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crops.append((y1, x1, y2, x2))
        return crops
    
    def slide_inference_decode(self, backbone, head, feature_list, crops, img_shape):
        """
        滑窗解码并融合（对齐 seg.py 的处理流程）
        
        Args:
            backbone: DINOv2 backbone（用于继续前向 + norm）
            head: 分割头
            feature_list: list of [1, 1+N, D] tensors（已量化）
            crops: list of (y1, x1, y2, x2)
            img_shape: (h_img, w_img)
        
        Returns:
            preds: [1, NUM_CLASSES, h_img, w_img] logits tensor（未 argmax）
        """
        h_img, w_img = img_shape
        h_crop, w_crop = self.CROP_SIZE
        patch_size = self.PATCH_SIZE
        
        preds = torch.zeros((1, self.NUM_CLASSES, h_img, w_img), device=self.device)
        count_mat = torch.zeros((1, 1, h_img, w_img), device=self.device)
        
        for i, (y1, x1, y2, x2) in enumerate(crops):
            feat = feature_list[i]  # [1, 1+N, D]
            if isinstance(feat, np.ndarray):
                feat = torch.from_numpy(feat).to(self.device)
            else:
                feat = feat.to(self.device)
            
            # 继续前向 + norm
            x = feat
            for blk_idx in range(self.layer_idx + 1, len(backbone.blocks)):
                x = backbone.blocks[blk_idx](x)
            x = backbone.norm(x)
            
            # 去掉 CLS token
            patch_tokens = x[:, 1:, :]  # [1, N, D]
            
            # 计算空间尺寸
            actual_h = y2 - y1
            actual_w = x2 - x1
            padded_h = math.ceil(actual_h / patch_size) * patch_size
            padded_w = math.ceil(actual_w / patch_size) * patch_size
            feat_h = padded_h // patch_size
            feat_w = padded_w // patch_size
            
            # Reshape → 分割头
            patch_tokens = patch_tokens.reshape(1, feat_h, feat_w, -1).permute(0, 3, 1, 2)
            logits = head(patch_tokens)  # [1, 21, feat_h, feat_w]
            
            # 上采样到 crop 尺寸，裁剪到实际尺寸
            logits_up = F.interpolate(logits, size=(h_crop, w_crop), mode='bilinear', align_corners=False)
            logits_crop = logits_up[:, :, :actual_h, :actual_w]
            
            # 累加
            preds[:, :, y1:y2, x1:x2] += logits_crop
            count_mat[:, :, y1:y2, x1:x2] += 1
        
        assert (count_mat == 0).sum() == 0, "count_mat 中有零值"
        preds = preds / count_mat
        return preds
    
    # ============== 评估主流程 ==============
    
    @torch.no_grad()
    def evaluate(self, seg_feat_dir, image_list=None, verbose=True):
        """
        执行分割评估
        
        Args:
            seg_feat_dir: 预提取特征目录（包含 name.npy 文件，格式 [num_slides, 1+N, D]）
            image_list: 图片名列表文件（每行一个名称，不含扩展名）
            verbose: 是否打印详细信息
        
        Returns:
            results: {
                'miou': mIoU,
                'acc': aAcc,
                'class_iou': 每类 IoU (numpy array),
            }
        """
        import mmcv
        from mmcv.parallel import collate
        from mmseg.datasets.pipelines import Compose
        from PIL import Image
        from tqdm import tqdm
        
        reg = _DINOV2_REGISTRY[self.model_name]

        cfg = mmcv.Config.fromfile(reg["config"])
        cfg.data_root = self.voc_root

        class _LoadImage:
            def __call__(self, results):
                results['filename'] = results['ori_filename'] = None
                img = results['img']
                results['img_shape'] = img.shape
                results['ori_shape'] = img.shape
                return results

        test_pipeline = Compose([_LoadImage()] + cfg.data.test.pipeline[1:])

        if verbose:
            print(f"  加载 backbone + 分割头 ({self.model_name})...")

        from dinov2.models import vision_transformer as vits
        vit_builder = getattr(vits, reg["vit_fn"])
        backbone = vit_builder(**reg["vit_kwargs"])
        backbone_ckpt = os.path.join(self.weights_root, reg["pretrain"])
        backbone.load_state_dict(torch.load(backbone_ckpt, map_location="cpu"), strict=True)
        backbone = backbone.to(self.device).eval()

        head_ckpt = os.path.join(self.weights_root, reg["seg_head"])
        head = load_seg_head(head_ckpt, in_channels=reg["embed_dim"],
                             num_classes=self.NUM_CLASSES, device=self.device)
        
        # 读取图片列表
        if image_list and os.path.exists(image_list):
            with open(image_list, 'r') as f:
                val_list = [ln.strip() for ln in f if ln.strip()]
        else:
            val_txt = os.path.join(self.voc_root, 'ImageSets/Segmentation/val.txt')
            with open(val_txt, 'r') as f:
                val_list = [ln.strip() for ln in f if ln.strip()]
        
        if verbose:
            print(f"  图片数: {len(val_list)}, 特征目录: {seg_feat_dir}")
            print("  开始 slide inference...")
        
        # 逐图评估
        hist = np.zeros((self.NUM_CLASSES, self.NUM_CLASSES), dtype=np.int64)
        missing = 0
        
        for name in tqdm(val_list, desc="Seg eval", disable=not verbose):
            # 检查特征文件
            feat_path = os.path.join(seg_feat_dir, f"{name}.npy")
            if not os.path.exists(feat_path):
                missing += 1
                continue
            
            # 加载图片 → pipeline 预处理 → 获取 img_shape
            img_path = os.path.join(self.voc_root, 'JPEGImages', f'{name}.jpg')
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)[:, :, ::-1]  # RGB → BGR
            
            data = test_pipeline(dict(img=img_np))
            data = collate([data], samples_per_gpu=1)
            h_img, w_img = data['img'][0].shape[2], data['img'][0].shape[3]
            
            # 计算 crops
            crops = self.get_slide_crops(h_img, w_img, self.CROP_SIZE, self.STRIDE)
            
            # 加载特征 [num_slides, 1+N, D]
            features = np.load(feat_path)
            assert features.shape[0] == len(crops), \
                f"{name}: slides 数不匹配 feat={features.shape[0]} vs crops={len(crops)}"
            
            # 对每个 slide 量化 + 校准
            quantized_list = []
            for s in range(features.shape[0]):
                tokens_np = features[s].astype(np.float32)  # [1+N, D]
                x_cal = self.quantize_tokens(tokens_np)      # [1+N, D] tensor
                quantized_list.append(x_cal.unsqueeze(0))     # [1, 1+N, D]
            
            # slide decode 融合（返回 logits tensor）
            preds_logits = self.slide_inference_decode(
                backbone, head, quantized_list, crops, (h_img, w_img)
            )
            
            # 加载 GT
            gt_path = os.path.join(self.voc_root, 'SegmentationClass', f'{name}.png')
            gt = np.array(Image.open(gt_path))
            ori_h, ori_w = gt.shape[:2]
            
            # bilinear 上采样 logits 到原图尺寸，再 argmax（对齐 seg.py）
            if (h_img, w_img) != (ori_h, ori_w):
                preds_logits = F.interpolate(
                    preds_logits, size=(ori_h, ori_w),
                    mode='bilinear', align_corners=False
                )
            seg_pred = preds_logits.argmax(dim=1).squeeze(0).cpu().numpy()
            
            # 更新混淆矩阵
            mask = gt != self.IGNORE_INDEX
            hist += np.bincount(
                self.NUM_CLASSES * gt[mask].astype(int) + seg_pred[mask].astype(int),
                minlength=self.NUM_CLASSES ** 2
            ).reshape(self.NUM_CLASSES, self.NUM_CLASSES)
        
        if missing > 0 and verbose:
            print(f"  [warn] 缺失特征: {missing} 张")
        
        # 计算指标（与 seg.py 一致，不加 epsilon，缺失类为 NaN 被 nanmean 忽略）
        iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        miou = np.nanmean(iou)
        acc = np.diag(hist).sum() / hist.sum()
        
        return {
            'miou': miou,
            'acc': acc,
            'class_iou': iou,
        }
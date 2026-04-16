"""
Evaluate an end-to-end compression model on an image dataset.
"""
import argparse
import json
import math
import sys
import time

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from pytorch_msssim import ms_ssim
from torchvision import transforms

import compressai

from compressai.ops import compute_padding
from compressai.zoo import image_models as pretrained_models
from compressai.zoo.image import model_architectures as architectures
from compressai.zoo.image_vbr import model_architectures as architectures_vbr
# zlt
import warnings
import re
import os
from compressai.datasets import MultiSourceFeatureFolder
from p2b_transform import p2b_inverse
import importlib.util
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
path = os.path.join(_PROJECT_ROOT, "coding", "preprocess", "dt_ufc", "nonlinear_transform_v2.py")
spec = importlib.util.spec_from_file_location("nonlinear_transform_v2", path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

nonlinear_quantization = mod.nonlinear_quantization
nonlinear_dequantization = mod.nonlinear_dequantization
#gcs
import numpy as np 
from compressai.datasets import FeatureFolder
import copy


torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)

# from torchvision.datasets.folder
IMG_EXTENSIONS = (
    ".npy",
)

architectures.update(architectures_vbr)
# zlt
def load_all_layer_stats(layer_stats_root, backbone_names):
    """
    layer_stats_paths: List[str] 与 backbone_names 一一对应
    每个 json 结构：{ "blk05": {"low":..., "high":...}, ... }
    """
    layer_stats = {}
    if isinstance(backbone_names, list):
        for backbone in backbone_names:
            model_stat_path = Path(layer_stats_root) / f"{backbone}.json"
            with open(model_stat_path, "r") as f:
                layer_stats[backbone] = json.load(f)
    else:
        model_stat_path = Path(layer_stats_root) / f"{backbone_names}.json"
        with open(model_stat_path, "r") as f:
            layer_stats[backbone_names] = json.load(f)

    return layer_stats

def load_all_quantization_mapping(quantization_mapping_root, backbone_names, layers, feat_transform):
    quantization_cache = {}
    if isinstance(backbone_names, list):
        for backbone in backbone_names:
            for l in layers:
                layer_mapping_path = Path(quantization_mapping_root) / backbone / f"{l}.json"
                with open(layer_mapping_path, "r") as f:
                    quantization_cache[(backbone, l)] = np.array(json.load(f))
    else:
        for l in layers:
            layer_mapping_path = Path(quantization_mapping_root) / backbone_names / f"{l}.json"
            with open(layer_mapping_path, "r") as f:
                quantization_cache[(backbone_names, l)] = np.array(json.load(f))
    return quantization_cache

def load_all_zscore_stats(stats_root, model_types, layers, device):
    cache = {}
    if isinstance(model_types, list):
        for model_name in model_types:
            for layer in layers:
                path = os.path.join(stats_root, f"zscore_{model_name}_{layer}.npz")
                data = np.load(path)
                mean = torch.from_numpy(data["mean"].astype(np.float32)).to(device)  # [C]
                std = torch.from_numpy(data["std"].astype(np.float32)).to(device)    # [C]
                cache[(model_name, layer)] = (mean, std)
    else:
        for layer in layers:
            path = os.path.join(stats_root, f"zscore_{model_types}_{layer}.npz")
            data = np.load(path)
            mean = torch.from_numpy(data["mean"].astype(np.float32)).to(device)  # [C]
            std = torch.from_numpy(data["std"].astype(np.float32)).to(device)    # [C]
            cache[(model_types, layer)] = (mean, std)
    return cache

def load_all_p2b_stats(stats_root: str,
                       model_type,
                       layers):
    cache = {}
    if isinstance(model_type, list):
        for model in model_type:
            for layer_name in layers:
                fname = f"p2b_{model}_{layer_name}.npz"
                path = os.path.join(stats_root, fname)
                if not os.path.isfile(path):
                    raise FileNotFoundError(f"P2B stats not found: {path}")

                with np.load(path) as data:
                    p = data["p"].astype(np.float32)
                    quantile_table = data["quantile_table"].astype(np.float32)

                cache[(model, layer_name)] = (p, quantile_table)
    else:
        for layer_name in layers:
            fname = f"p2b_{model}_{layer_name}.npz"
            path = os.path.join(stats_root, fname)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"P2B stats not found: {path}")

            with np.load(path) as data:
                p = data["p"].astype(np.float32)
                quantile_table = data["quantile_table"].astype(np.float32)

            cache[(model, layer_name)] = (p, quantile_table)

    return cache


def collect_eval_files(args):
    """收集所有待评测文件路径"""
    if args.layers and args.backbone_root:
        # 多层模式：合并所有层的 npy 文件
        filelist = []

        layer_dirs = [Path(args.backbone_root) / args.backbone_name / l for l in args.layers]
        for d in layer_dirs:
            if not d.is_dir():
                raise RuntimeError(f"Missing layer dir: {d}")
            filelist.extend(sorted(d.glob("*.npy")))
        if not filelist:
            raise RuntimeError("No .npy files found in provided layers.")
        return filelist
    else:
        # 默认单层
        d = Path(args.dataset)
        return sorted(d.glob("*.npy"))


def collect_images(rootpath: str) -> List[str]:
    image_files = []

    for ext in IMG_EXTENSIONS:
        image_files.extend(Path(rootpath).rglob(f"*{ext}"))
    return sorted(image_files)


def psnr(a: torch.Tensor, b: torch.Tensor, max_val: int = 255) -> float:
    return 20 * math.log10(max_val) - 10 * torch.log10((a - b).pow(2).mean())

#gcs, note that the mse computed here does not consider the clip operation, this is not the actual mse
def compute_metrics(
    org: torch.Tensor, rec: torch.Tensor, trun_low: float = -5.0, trun_high: float = 5.0
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {} 
    interval = trun_high - trun_low
    org = (org * interval + trun_low).clamp(trun_low, trun_high)  # verified, exactly same as original feature before scaling but after clip!!!
    # print('scale', [f"{x:.8f}" for x in org[0, 0, 0, :10]])
    rec = (rec * interval + trun_low).clamp(trun_low, trun_high)
    metrics["mse"] = (org - rec).pow(2).mean().item()   #verified, exactly same as the numpy computation in mse.py
    return metrics


@torch.no_grad()
#gcs
def inference(model, x, vbr_stage=None, vbr_scale=None):
    # x = x.unsqueeze(0)    #gcs, no need for feature, already [N,C,H,W] 4 dims

    h, w = x.size(2), x.size(3)
    pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2

    x_padded = F.pad(x, pad, mode="constant", value=0)

    start = time.time()
    out_enc = (
        model.compress(x_padded)
        if vbr_scale is None
        else model.compress(x_padded, stage=vbr_stage, s=0, inputscale=vbr_scale)
    )
    enc_time = time.time() - start

    start = time.time()
    out_dec = (
        model.decompress(out_enc["strings"], out_enc["shape"])
        if vbr_scale is None
        else model.decompress(
            out_enc["strings"],
            out_enc["shape"],
            stage=vbr_stage,
            s=0,
            inputscale=vbr_scale,
        )
    )
    dec_time = time.time() - start

    out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)

    # input images are 8bit RGB for now
    num_points = x.size(0) * x.size(1) * x.size(2) * x.size(3)
    bpfp = sum(len(s[0]) for s in out_enc["strings"]) * 8.0 / num_points

    #gcs, return rec_feat
    rec_feat = out_dec["x_hat"]
    return rec_feat, {
        #gcs
        "bpfp": bpfp,
        "encoding_time": enc_time,
        "decoding_time": dec_time,
    }


@torch.no_grad()
def inference_entropy_estimation(model, x, vbr_stage=None, vbr_scale=None):
    x = x.unsqueeze(0)

    start = time.time()
    out_net = (
        model.forward(x)
        if vbr_scale is None
        else model.forward(x, stage=vbr_stage, inputscale=vbr_scale)
    )
    elapsed_time = time.time() - start

    # input images are 8bit RGB for now
    metrics = compute_metrics(x, out_net["x_hat"], 255)
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp = sum(
        (torch.log(likelihoods).sum() / (-math.log(2) * num_pixels))
        for likelihoods in out_net["likelihoods"].values()
    )

    return {
        "psnr-rgb": metrics["psnr-rgb"],
        "ms-ssim-rgb": metrics["ms-ssim-rgb"],
        "bpp": bpp.item(),
        "encoding_time": elapsed_time / 2.0,  # broad estimation
        "decoding_time": elapsed_time / 2.0,
    }


def load_pretrained(model: str, metric: str, quality: int) -> nn.Module:
    return pretrained_models[model](
        quality=quality, metric=metric, pretrained=True, progress=False
    ).eval()


def load_checkpoint(arch: str, no_update: bool, checkpoint_path: str) -> nn.Module:
    # update model if need be
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint
    # compatibility with 'not updated yet' trained nets
    for key in ["network", "state_dict", "model_state_dict"]:
        if key in checkpoint:
            state_dict = checkpoint[key]
    
    #gcs, remove "module."
    # if arch == "bmshj2018-hyperprior":
    #     prefix = "module."
    #     state_dict = {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}
    # print(state_dict.keys())
    model_cls = architectures[arch]
    if arch in ["bmshj2018-hyperprior-vbr", "mbt2018-mean-vbr"]:
        net = model_cls.from_state_dict(state_dict, vr_entbttlnck=True)
        if not no_update:
            net.update(force=True, scale=net.Gain[-1])
    else:
        net = model_cls.from_state_dict(state_dict)
        if not no_update:
            net.update(force=True)

    return net.eval()


def eval_model(
    model: nn.Module,
    outputdir: Path,
    inputdir: Path,
    filepaths,
    entropy_estimation: bool = False,
    trained_net: str = "",
    description: str = "",
    vbr_stage=None,
    vbr_scale=None,
    **args: Any,
) -> Tuple[defaultdict[Any, float], Dict[Any, dict]]:
    device = next(model.parameters()).device

    feat_transform = args["feat_transform"]

    if feat_transform == "p2b":
        p2b_cache = load_all_p2b_stats(
            stats_root=args['p2b_stats_root'],
            model_type=args['backbone_name'],
            layers=args['layers'],
        )
    elif feat_transform == "zscore":
        zscore_cache = load_all_zscore_stats(
            stats_root=args["zscore_stats_root"],
            model_types=args["backbone_name"],
            layers=args["layers"],
            device=device,
        )
    elif feat_transform == "trunc" or feat_transform == "residual_trunc":
        layer_stats_cache = load_all_layer_stats(args["layer_stats_root"], args["backbone_name"])
    else: # 'kmeans' / 'density' / 'blend' / 'ekmeans'
        kmeans_bits = int(re.search(r'_(\d+)bit$', feat_transform).group(1))
        quantization_cache = load_all_quantization_mapping(args["quantization_mapping_root"], args["backbone_name"], args["layers"], feat_transform)
        layer_stats_cache = load_all_layer_stats(args["layer_stats_root"], args["backbone_name"])

    metrics = defaultdict(float)
    is_vbr_model = args["architecture"].endswith("-vbr")
    # zlt 新增：逐层累加器 + 计数器
    per_layer_sums = defaultdict(lambda: defaultdict(float))
    per_layer_counts = defaultdict(int)
    #gcs
    model_type = args['backbone_name']; task = args["task"]; trun_flag = args["trun_flag"]; trun_low = args["trun_low"]
    trun_high = args['trun_high']; quant_type = args['quant_type']; qsamples = args['qsamples']; bit_depth = args['bit_depth']

    for filepath in filepaths:
        # feat = np.load(filepath).astype(np.float32)
        # org_feat = copy.deepcopy(feat)
        layer_name = Path(filepath).parent.name

        # 原始特征始终从原 test 集读取
        org_feat = np.load(filepath).astype(np.float32)   # [257, C]

        # === NEW: 根据 feat_transform 选择输入给 codec 的特征 ===
        if feat_transform == "p2b":
            p2b_feat_path = Path(args['preprocess_root']).joinpath(*filepath.parts[-3:])
            feat = np.load(p2b_feat_path).astype(np.float32)
            # packing
            feat = MultiSourceFeatureFolder.feat_forward_transform(feat, n_split=64)
            
        elif feat_transform == "zscore":
            mean_t, std_t = zscore_cache[(model_type, layer_name)]
            mean = mean_t.cpu().numpy()
            std = std_t.cpu().numpy()
            std_safe = np.maximum(std, 1e-6)
            feat = (org_feat - mean[None, :]) / std_safe[None, :]
            # packing
            feat = MultiSourceFeatureFolder.feat_forward_transform(feat, n_split=64)
            
        elif feat_transform == "trunc" or feat_transform == "residual_trunc":
            assert trun_flag is True
            # 读取 per-layer low/high
            # low = float(layer_stats_cache[args['backbone_name']][layer_name]["low"])
            # high = float(layer_stats_cache[args['backbone_name']][layer_name]["high"])
            # feat = FeatureFolder.truncation(org_feat, low, high)
            # q_low, q_high = low, high
            # feat = FeatureFolder.uniform_quantization(feat, q_low, q_high, bit_depth)
            min_val = np.min(org_feat)
            max_val = np.max(org_feat)   
            feat = (org_feat - min_val) / (max_val - min_val)
            # packing
            feat = MultiSourceFeatureFolder.feat_forward_transform(feat, n_split=64)
            
        else: # 'kmeans' / 'density' / 'blend' / 'ekmeans'
            low = float(layer_stats_cache[args['backbone_name']][layer_name]["low"])
            high = float(layer_stats_cache[args['backbone_name']][layer_name]["high"])
            feat = FeatureFolder.truncation(org_feat, low, high)
            # # packing
            feat = MultiSourceFeatureFolder.feat_forward_transform(feat)  # [H, W]
            kmeans_points = quantization_cache[(args['backbone_name'], layer_name)]
            feat = nonlinear_quantization(feat, kmeans_points, kmeans_bits)
            
        x = torch.from_numpy(feat).to(device)
        x = x[None, None, :, :]

        #gcs, load feature in the same way of training
        # feat = np.load(filepath)
        # org_feat = copy.deepcopy(feat)
        # N, C, H, W = feat.shape
        #gcs, preprocessing
        # if not args["half"]: feat = feat.astype(np.float32)
        # if trun_flag == True: feat = FeatureFolder.truncation(feat, trun_low, trun_high)
        # feat = FeatureFolder.uniform_quantization(feat, trun_low, trun_high, bit_depth)
        # feat = FeatureFolder.packing(feat, model_type)
        # x = torch.from_numpy(feat).to(device)
        # x = x.unsqueeze(0); x = x.unsqueeze(0) # reshape to [1,1,H,W]

        if not entropy_estimation:
            if args["half"]:
                model = model.half()
                x = x.half()
            #gcs, return rec_feat
            rec_feat, rv = (
                        inference(model, x)
                        if not is_vbr_model
                        else inference(model, x, vbr_stage, vbr_scale)
            )
            # zlt, postprocessing
            rec_feat = rec_feat.squeeze((0, 1))
            rec_feat = rec_feat.cpu().detach().numpy()


            # === NEW: 按 transform 做逆变换回原空间 ===
            if feat_transform == "p2b":
                rec_feat = MultiSourceFeatureFolder.feat_inverse_transform(rec_feat, n_split=64)  # [257, C]
                p, quantile_table = p2b_cache[(model_type, layer_name)]
                rec_feat = np.expand_dims(rec_feat, axis=0)
                rec_feat = p2b_inverse(rec_feat, p, quantile_table).squeeze(0)
            elif feat_transform == "zscore":
                rec_feat = MultiSourceFeatureFolder.feat_inverse_transform(rec_feat, n_split=64)  # [257, C]
                mean_t, std_t = zscore_cache[(model_type, layer_name)]
                mean = mean_t.cpu().numpy()
                std = std_t.cpu().numpy()
                rec_feat = rec_feat * std[None, :] + mean[None, :]
            elif feat_transform == "trunc" or feat_transform == "residual_trunc":
                rec_feat = MultiSourceFeatureFolder.feat_inverse_transform(rec_feat, n_split=64)  # [257, C]
                # rec_feat = FeatureFolder.uniform_dequantization(rec_feat, q_low, q_high, bit_depth)
                rec_feat = rec_feat * (max_val - min_val) + min_val
            else: # 'kmeans' / 'density' / 'blend' / 'ekmeans'
                kmeans_points = quantization_cache[(args['backbone_name'], layer_name)]
                rec_feat = nonlinear_dequantization(rec_feat, kmeans_points, kmeans_bits)
                rec_feat = MultiSourceFeatureFolder.feat_inverse_transform(rec_feat, n_split=64)  # [257, C]

            dtype = np.float32
            #gcs, postprocessing
            # rec_feat = rec_feat.squeeze(0); rec_feat = rec_feat.squeeze(0)
            # rec_feat = rec_feat.cpu().detach().numpy()
            # rec_feat = FeatureFolder.unpacking(rec_feat, [N, C, H, W], model_type)
            # rec_feat = FeatureFolder.uniform_dequantization(rec_feat, trun_low, trun_high, bit_depth)
            # Feature dtype
            # dtype = np.float16 if model_type=='sd3' else np.float32
            rec_feat = rec_feat.astype(dtype)
            # Compute MSE
            feat_mse = np.mean((rec_feat-org_feat)**2)
            rv["mse"] = feat_mse
        else:
            rv = (
                inference_entropy_estimation(model, x)
                if not is_vbr_model
                else inference_entropy_estimation(model, x, vbr_stage, vbr_scale)
            )
        for k, v in rv.items():
            metrics[k] += v
        # zlt
        per_layer_counts[layer_name] += 1
        for k, v in rv.items():
            per_layer_sums[layer_name][k] += v
        # end
        if args["per_image"]:
            if not Path(outputdir).is_dir():
                raise FileNotFoundError("Please specify output directory")

            output_subdir = Path(outputdir) / Path(filepath).parent.relative_to(
                inputdir
            )
            output_subdir.mkdir(parents=True, exist_ok=True)
            #gcs, save rec_feat
            rec_feat_name = output_subdir / f"{filepath.stem}.npy"
            # zlt
            # 去掉前两维度 (1,1,H,W) → (H,W)
            if isinstance(rec_feat, np.ndarray) and rec_feat.ndim == 4 and rec_feat.shape[0] == 1 and rec_feat.shape[
                1] == 1:
                rec_feat = rec_feat.squeeze((0, 1))

            np.save(rec_feat_name, rec_feat.astype(np.float32))

    for k, v in metrics.items():
        metrics[k] = v / len(filepaths)
    # zlt
    by_layer = {}
    for lname, sums in per_layer_sums.items():
        cnt = max(1, per_layer_counts[lname])
        by_layer[lname] = {k: sums[k] / cnt for k in sums.keys()}

    return metrics, by_layer
    # return metrics

# Custom type parsing function
def parse_truncation(value):
    try:
        # Try to parse as a float
        return float(value)
    except ValueError:
        # Try to parse as a list of floats (comma-separated)
        try:
            return [float(x) for x in value.strip('[]').split(',')]
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Invalid input for --truncation: {value}. Must be a float or a list of floats."
            )

def setup_args():
    # Common options.
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("dataset", type=str, help="dataset path")
    parent_parser.add_argument(
        "-a",
        "--architecture",
        type=str,
        choices=pretrained_models.keys(),
        help="model architecture",
        required=True,
    )
    parent_parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parent_parser.add_argument(
        "--cuda",
        action="store_true",
        help="enable CUDA",
    )
    parent_parser.add_argument(
        "--half",
        action="store_true",
        help="convert model to half floating point (fp16)",
    )
    parent_parser.add_argument(
        "--entropy-estimation",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parent_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose mode",
    )
    parent_parser.add_argument(
        "-m",
        "--metric",
        type=str,
        choices=["mse", "ms-ssim"],
        default="mse",
        help="metric trained against (default: %(default)s)",
    )
    parent_parser.add_argument(
        "-d",
        "--output_directory",
        type=str,
        default="",
        help="path of output directory. Optional, required for output json file, results per image. Default will just print the output results.",
    )
    #gcs, add model_type, trun_flag, trun_low, trun_high, quant_type, qsamples, bit_depth
    # parent_parser.add_argument(
    #     "-model_type",
    #     "--model_type",
    #     type=str,
    #     default="sd3",
    #     help="Please input the model_type.",
    # )
    parent_parser.add_argument(
        "-task",
        "--task",
        type=str,
        default="tti",
        help="Please input the task.",
    )
    parent_parser.add_argument(
        "-trun_flag",
        "--trun_flag",
        type=bool,
        default=True,
        help="Please input the trun_flag.",
    )
    parent_parser.add_argument(
        "-trun_low",
        "--trun_low",
        type=parse_truncation,
        default=-5,
        help="Please input the truncated upper value (float or list of floats).",
    )
    parent_parser.add_argument(
        "-trun_high",
        "--trun_high",
        type=parse_truncation,
        default=5,
        help="Please input the truncated upper value (float or list of floats).",
    )
    parent_parser.add_argument(
        "-quant_type",
        "--quant_type",
        type=str,
        default="uniform",
        help="Please input the quant_type.",
    )
    parent_parser.add_argument(
        "-qsamples",
        "--qsamples",
        type=int,
        default=0,
        help="Please input the qsamples.",
    )
    parent_parser.add_argument(
        "-bit_depth",
        "--bit_depth",
        type=int,
        default=1,
        help="Please input the bit_depth.",
    )
    parent_parser.add_argument(
        "-o",
        "--output-file",
        type=str,
        default="",
        help="output json file name, (default: architecture-entropy_coder.json)",
    )
    parent_parser.add_argument(
        "--per-image",
        action="store_true",
        help="store results for each image of the dataset, separately",
    )
    # Options for variable bitrate (vbr) models
    parent_parser.add_argument(
        "--vbr_quantstep",
        dest="vbr_quantstepsizes",
        type=str,
        default="10.0000,7.1715,5.1832,3.7211,2.6833,1.9305,1.3897,1.0000",
        help="Quantization step sizes for variable bitrate (vbr) model. Floats [10.0 , 1.0] (example: 10.0,8.0,6.0,3.0,1.0)",
    )
    parent_parser.add_argument(
        "--vbr_tr_stage",
        type=int,
        choices=[1, 2],
        default=2,
        help="Stage in vbr model training. \
            1: Model behaves/runs like a regular single-rate \
            model without using any vbr tool (use for training/testing a model for single/highest lambda). \
            2: Model behaves/runs like a vbr model using vbr tools. (use for post training stage=1 result.)",
    )
    # zlt
    parent_parser.add_argument(
        "--backbone_root", type=str, default=None,
        help="Root under /features/{split}, e.g. /path/to/features/test"
    )
    parent_parser.add_argument(
        "--backbone_name", type=str, default="dinov2_vitl14",
        help="Backbone subfolder name under split/, e.g. dinov2_vitl14"
    )
    parent_parser.add_argument(
        "--layers", type=str, nargs="*", default=None,
        help="List of feature layers, e.g. blk05 blk11 blk17 blk23"
    )
    parent_parser.add_argument(
        "--layer_stats_root", type=str, default=None,
        help="Per-layer global min/max (json)"
    )
    parent_parser.add_argument(
        "--preprocess_root", type=str, default=None,
        help="preprocessed test dataset root"
    )
    parent_parser.add_argument(
        "--p2b_stats_root", type=str, default=None,
        help="p2b stat npz root"
    )
    parent_parser.add_argument(
        "--feat_transform",
        type=str,
        default="p2b",
        help="Feature pre-transform used before codec",
    )
    parent_parser.add_argument(
        "--zscore_stats_root",
        type=str,
        default=None,
        help="Root directory of zscore stats (npz)",
    )
    parent_parser.add_argument(
        "--quantization_mapping_root",
        type=str,
        default=None,
        help="Root directory of quantization mapping (json)",
    )
    parent_parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Output json file to save the overall results.",
    )
    
    parser = argparse.ArgumentParser(
        description="Evaluate a model on an image dataset.", add_help=True
    )
    subparsers = parser.add_subparsers(help="model source", dest="source")

    # Options for pretrained models
    pretrained_parser = subparsers.add_parser("pretrained", parents=[parent_parser])
    pretrained_parser.add_argument(
        "-q",
        "--quality",
        dest="qualities",
        type=str,
        default="1",
        help="Pretrained model qualities. (example: '1,2,3,4') (default: %(default)s)",
    )

    checkpoint_parser = subparsers.add_parser("checkpoint", parents=[parent_parser])
    checkpoint_parser.add_argument(
        "-p",
        "--path",
        dest="checkpoint_paths",
        type=str,
        nargs="*",
        required=True,
        help="checkpoint path",
    )
    checkpoint_parser.add_argument(
        "--no-update",
        action="store_true",
        help="Disable the default update of the model entropy parameters before eval",
    )

    return parser


def main(argv):  # noqa: C901
    parser = setup_args()
    args = parser.parse_args(argv)

    if args.source not in ["checkpoint", "pretrained"]:
        print("Error: missing 'checkpoint' or 'pretrained' source.", file=sys.stderr)
        parser.print_help()
        raise SystemExit(1)

    description = (
        "entropy-estimation" if args.entropy_estimation else args.entropy_coder
    )

    # filepaths = collect_images(args.dataset)
    # zlt
    filepaths = collect_eval_files(args)
    if len(filepaths) == 0:
        print("Error: no images found in directory.", file=sys.stderr)
        raise SystemExit(1)

    compressai.set_entropy_coder(args.entropy_coder)

    is_vbr_model = args.architecture.endswith("-vbr")

    # create output directory
    if args.output_directory:
        Path(args.output_directory).mkdir(parents=True, exist_ok=True)

    if args.source == "pretrained":
        args.qualities = [int(q) for q in args.qualities.split(",") if q]
        runs = sorted(args.qualities)
        opts = (args.architecture, args.metric)
        if is_vbr_model:
            opts += (0,)
        load_func = load_pretrained
        log_fmt = "\rEvaluating {0} | {run:d} "
    else:
        runs = args.checkpoint_paths
        opts = (args.architecture, args.no_update)
        if is_vbr_model:
            opts += (args.checkpoint_paths[0],)
        load_func = load_checkpoint
        log_fmt = "\rEvaluating {run:s} "

    if is_vbr_model:
        if args.source == "checkpoint":
            assert (
                len(args.checkpoint_paths) <= 1
            ), "Use only one checkpoint for vbr model."
        scales = [1.0 / float(q) for q in args.vbr_quantstepsizes.split(",") if q]
        runs = sorted(scales)
        runs = torch.tensor(runs)
        log_fmt = "\rEvaluating quant step {run:5.2f} "
        model = load_func(*opts)
        # set some arch specific params for vbr
        model.no_quantoffset = False
        if args.architecture in ["mbt2018-vbr"]:
            model.scl2ctx = True
        if args.cuda and torch.cuda.is_available():
            model = model.to("cuda")
            runs = runs.to("cuda")

    results = defaultdict(list)
    # zlt
    results_by_layer = defaultdict(lambda: defaultdict(list))
    # end
    for run in runs:
        if args.verbose:
            sys.stderr.write(
                log_fmt.format(*opts, run=(run if not is_vbr_model else 1.0 / run))
            )
            sys.stderr.flush()
        if not is_vbr_model:
            model = load_func(*opts, run)
        else:
            # update bottleneck for every new quant_step if vbr bottleneck is used in the model
            if (
                args.architecture in ["bmshj2018-hyperprior-vbr", "mbt2018-mean-vbr"]
                and args.vbr_tr_stage == 2
            ):
                model.update(force=True, scale=run)
        if args.source == "pretrained":
            trained_net = f"{args.architecture}-{args.metric}-{run}-{description}"
        else:
            run_ = run if not is_vbr_model else args.checkpoint_paths[0]
            cpt_name = Path(run_).name[: -len(".pth.tar")]  # removesuffix() python3.9
            trained_net = f"{cpt_name}-{description}"
        print(f"Using trained model {trained_net}", file=sys.stderr)
        if args.cuda and torch.cuda.is_available() and not is_vbr_model:
            model = model.to("cuda")
        args_dict = vars(args)

        # 多层模式下，用 backbone_root 作为“输入根路径”；否则用 dataset
        if args.layers and args.backbone_root:
            inputdir_root = str(Path(args.backbone_root) / args.backbone_name)  # 仅到 dinov2_vitl14 这一层
        else:
            inputdir_root = args.dataset

        metrics, by_layer = eval_model(
            model,
            args.output_directory,
            inputdir_root,
            filepaths,
            trained_net=trained_net,
            description=description,
            vbr_stage=None if not is_vbr_model else args.vbr_tr_stage,
            vbr_scale=None if not is_vbr_model else run,
            **args_dict,
        )
        # metrics = eval_model(
        #     model,
        #     args.output_directory,
        #     args.dataset,
        #     filepaths,
        #     trained_net=trained_net,
        #     description=description,
        #     vbr_stage=None if not is_vbr_model else args.vbr_tr_stage,
        #     vbr_scale=None if not is_vbr_model else run,
        #     **args_dict,
        # )
        for k, v in metrics.items():
            results[k].append(v)
        # zlt 新增：把逐层结果也写入容器（按 run 维度追加）
        for lname, mdict in by_layer.items():
            for k, v in mdict.items():
                results_by_layer[lname][k].append(v)

    if args.verbose:
        sys.stderr.write("\n")
        sys.stderr.flush()

    description = (
        "entropy estimation" if args.entropy_estimation else args.entropy_coder
    )
    # zlt
    output = {
        "name": f"{args.architecture}-{args.metric}",
        "description": f"Inference ({description})",
        "results_overall": results,  # 原有总体，多 run 时是列表
        "results_by_layer": results_by_layer  # 新增分层，多 run 时每层是列表
    }

    # output = {
    #     "name": f"{args.architecture}-{args.metric}",
    #     "description": f"Inference ({description})",
    #     "results": results,
    # }
    if args.output_directory:
        output_file = (
            args.output_file
            if args.output_file
            else f"{args.architecture}-{description}"
        )

        with (Path(f"{args.output_directory}/{output_file}").with_suffix(".json")).open(
            "wb"
        ) as f:
            f.write(json.dumps(output, indent=2).encode())

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])

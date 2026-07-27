#!/usr/bin/env python
"""
ORFC-v1.1 experiment runner.

Extends ORFC-v1 with:
  - Response objective selection: legacy / fixed_energy / operational (§15.4)
  - Multi-alpha single-sided probes (§15.3)
  - Per-image energy matching (§15.3.1)
  - Pairwise dispersion loss (§15.4)
  - D̄₀ = D_tail/(T·D) per-element normalisation (§15.2)
  - K256 high-rate verification (§15.6)
  - Comprehensive S_g, M_g, N_g, q_g, κ_g held-out metrics

All results are saved under a separate run_id.  This script NEVER
writes to ``formal_k8_20260726T175547Z``.

Usage:
    # Standard OPQ baseline:
    python run_v1_1.py --layer blk20 --K 8 --response_objective none --beta 0

    # Fixed-energy joint:
    python run_v1_1.py --layer blk20 --K 8 --response_objective fixed_energy \\
                       --beta 0.1

    # Operational joint:
    python run_v1_1.py --layer blk20 --K 8 --response_objective operational \\
                       --beta 0.1

    # Legacy joint (v1 reproduction):
    python run_v1_1.py --layer blk20 --K 8 --response_objective legacy \\
                       --beta 0.03

    # K256 high-rate:
    python run_v1_1.py --layer blk20 --K 256 --response_objective fixed_energy \\
                       --beta 0.1
"""

import os, sys, argparse, json, math, time
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

V1_ROOT = os.path.dirname(os.path.abspath(__file__))
ORFC_ROOT = os.path.normpath(os.path.join(V1_ROOT, '..', 'orfc'))
PROJECT_ROOT = os.path.normpath(os.path.join(V1_ROOT, '..', '..', '..'))
CODE_VERSION = 'orfcv1.1-20260726'

PROTECTED_RUN_IDS = frozenset({
    'formal_k8_20260726T175547Z',
})

if ORFC_ROOT not in sys.path:
    sys.path.insert(0, ORFC_ROOT)
if V1_ROOT not in sys.path:
    sys.path.insert(0, V1_ROOT)

from run_multilayer_calibrator import (
    set_seed, preload_features, load_gt, evaluate_accuracy,
)
from opq import (
    batch_normalize_gpu, batch_inv_normalize_gpu, batched_assign,
    learn_opq_rotation,
)
from backbone.wrapper import Dinov2Wrapper
from soft_pq import (
    SoftPQ, OrthogonalTransform, FeatureTransform,
    FrozenTail,
)

from data_utils import (
    make_split, save_manifest, make_seeds,
    build_result_skeleton, git_status_porcelain,
)
from codec_v1 import (
    FeatureCodecV1, reconstruction_audit,
    save_codec_v1, load_codec_v1,
)
from train_v1 import train_v1, evaluate_heldout_v1_1
from run_v1 import (
    pq_encode_decode_features,
    evaluate_delta_l_ref, evaluate_rate,
    codec_encode_decode,
    _opq_artifact_path, save_opq_artifact, load_opq_artifact,
    _opq_lock_path, _acquire_opq_lock, _release_opq_lock,
    check_cayley_init, _codec_labels, _histogram_pmf,
    _cache_dir, _features_cache_path, _teacher_cache_path,
    _save_array_atomic, _load_or_create_array, _compute_teacher_cache,
)

import warnings
warnings.filterwarnings("ignore", message="xFormers is available")
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")
warnings.filterwarnings("ignore", message="numpy.ndarray size changed")
import logging
try:
    from mmcv.utils import get_logger
    logger = get_logger('mmcv')
except ImportError:
    logger = logging.getLogger('mmcv')
logger.setLevel(logging.WARNING)


def _result_json_atomic(path, data):
    """Atomically write JSON result file."""
    import tempfile
    d = os.path.dirname(path) or '.'
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix='.json.tmp')
    os.close(fd)
    try:
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def _load_test_reference(path, args, manifest, expected_checkpoint=None):
    """Load a frozen test result after validating its evaluation contract."""
    with open(path) as f:
        data = json.load(f)
    cfg = data.get('config', {})
    expected = {
        'layer': args.layer,
        'K': args.K,
        'embedding_dim': args.embedding_dim,
        'norm_mode': args.norm_mode,
        'seed': args.seed,
    }
    mismatches = {
        key: (cfg.get(key), value)
        for key, value in expected.items()
        if cfg.get(key) != value
    }
    ref_manifest = data.get('split_manifest', {})
    if ref_manifest.get('test_basenames') != manifest.get('test_basenames'):
        mismatches['test_basenames'] = ('reference', 'current')
    if expected_checkpoint is not None:
        reference_checkpoint = data.get('checkpoint')
        if (not reference_checkpoint
                or os.path.realpath(reference_checkpoint)
                != os.path.realpath(expected_checkpoint)):
            mismatches['checkpoint'] = (
                reference_checkpoint, expected_checkpoint)
    if mismatches:
        raise ValueError(
            f"Reference result contract mismatch for {path}: {mismatches}")

    test = data.get('test', {})
    accuracy = test.get('accuracy', data.get('std_opq_acc'))
    if accuracy is None:
        raise ValueError(f"Reference result has no test accuracy: {path}")
    return {
        'accuracy': float(accuracy),
        'D0': test.get(
            'D0', data.get('std_opq_D0_raw', data.get('v1_1_D0_raw'))),
        'path': os.path.realpath(path),
        'checkpoint': data.get('checkpoint'),
        'method': data.get('method'),
    }


def _opq_artifact_path_v1_1(args, seeds):
    """Deterministic OPQ artifact path (includes K in name for K256)."""
    art_dir = os.path.join(V1_ROOT, 'artifacts', args.backbone)
    os.makedirs(art_dir, exist_ok=True)
    name = (f"opq_{args.layer}_K{args.K}_emb{args.embedding_dim}"
            f"_{args.norm_mode}_n{args.n_train}"
            f"_ss{seeds['split']}_is{seeds['init']}"
            f"_oi{args.opq_iter}_ki{args.kmeans_iter}"
            f"_km{args.kmeans_max_samples}"
            f"_{CODE_VERSION}.npz")
    return os.path.join(art_dir, name)


# ================================================================
#  Main experiment
# ================================================================

def run_experiment(args):
    # Protect old results
    if args.run_id in PROTECTED_RUN_IDS:
        raise ValueError(
            f"run_id '{args.run_id}' is protected. "
            f"ORFC-v1.1 must not overwrite v1 results.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    layer_idx = int(args.layer[-2:])
    seeds = make_seeds(args.seed)
    set_seed(seeds['base'])

    D = 1024
    bt_dim = args.bottleneck_dim
    Dp = bt_dim if bt_dim > 0 else D
    num_groups = Dp // args.embedding_dim
    bits_per_token = num_groups * math.log2(args.K)

    # Phase tag
    if args.freeze_codebooks and not args.freeze_transform:
        phase_tag, phase_desc = "A", "U only"
    elif args.step_mode == 'alternating':
        phase_tag, phase_desc = "C", "alternating"
    else:
        phase_tag, phase_desc = "B", "joint"

    # Alpha list
    alpha_list = [float(a) for a in args.alpha_list.split(',')]
    alpha_weights = None
    if args.alpha_weights:
        alpha_weights = [float(w) for w in args.alpha_weights.split(',')]
        if len(alpha_weights) != len(alpha_list):
            raise ValueError(
                f"alpha_weights length {len(alpha_weights)} != "
                f"alpha_list length {len(alpha_list)}")

    response_objective = args.response_objective
    if alpha_weights is None:
        if response_objective == 'operational':
            # M(alpha=1) is the registered primary operational marginal.
            alpha_weights = [
                1.0 if a == max(alpha_list) else 0.0
                for a in alpha_list]
        else:
            alpha_weights = [1.0 / len(alpha_list)] * len(alpha_list)
    _use_elastic = (
        (args.beta > 0 or args.beta_target_ratio > 0)
        and response_objective != 'none')

    print(f"\n{'#' * 70}")
    print(f"# ORFC-v1.1 Experiment")
    print(f"# layer={args.layer}, K={args.K}, G={num_groups}, "
          f"d={args.embedding_dim}")
    print(f"# response_objective={response_objective}")
    print(f"# β={args.beta}, alphas={alpha_list}, "
          f"weights={alpha_weights}")
    print(f"# D0 normalisation: {args.d0_normalize}")
    print(f"# bits/token={bits_per_token:.0f}")
    print(f"# Phase: {phase_tag} ({phase_desc})")
    print(f"# seeds: {seeds}")
    print(f"# {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'#' * 70}")

    # ---- Load features ----
    train_dir = (Path(args.feat_root)
                 / args.train_subset / args.backbone / args.layer)
    test_dir = (Path(args.feat_root)
                / args.test_subset / args.backbone / args.layer)
    train_files = sorted(train_dir.glob("*.npy"))
    test_files = sorted(test_dir.glob("*.npy"))
    if args.n_test > 0:
        test_files = test_files[:args.n_test]
    if not train_files:
        raise FileNotFoundError(f"No train features in {train_dir}")
    if not test_files:
        raise FileNotFoundError(f"No test features in {test_dir}")

    print(f"\nData: train_pool={len(train_files)}, "
          f"test={len(test_files)}")

    # ---- Deterministic split ----
    split, manifest = make_split(
        train_files, test_files,
        n_train=args.n_train, n_val=args.n_val,
        split_seed=seeds['split'],
    )

    out_dir = os.path.join(
        V1_ROOT, 'results', args.backbone, args.run_id)
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(
        out_dir, f'split_manifest_s{args.seed}.json')
    save_manifest(manifest, manifest_path)
    print(f"  Split: train={len(split['train'])}, "
          f"val={len(split['val'])}, test={len(split['test'])}")

    # ---- Load features (shared cache) ----
    gt_test = None if args.skip_test_eval else load_gt(args.gt_path)

    feat_train_path = _features_cache_path(
        args, seeds, 'train', args.n_train)
    feat_val_path = _features_cache_path(
        args, seeds, 'val', args.n_val)

    if os.path.isfile(feat_train_path) and os.path.isfile(feat_val_path):
        print(f"  Loading cached stacked features (mmap)...")
        features_train_array = np.load(feat_train_path, mmap_mode='r')
        features_val_array = np.load(feat_val_path, mmap_mode='r')
        if args.skip_test_eval:
            features_test, basenames_test = None, []
        else:
            features_test, basenames_test = preload_features(
                split['test'], num_workers=4)
    else:
        features_train, _ = preload_features(
            split['train'], num_workers=4)
        features_val, _ = preload_features(
            split['val'], num_workers=4)
        if args.skip_test_eval:
            features_test, basenames_test = None, []
        else:
            features_test, basenames_test = preload_features(
                split['test'], num_workers=4)

        features_train_array = np.stack(features_train)
        features_val_array = np.stack(features_val)
        del features_train, features_val

        print(f"  Saving stacked features to cache...")
        _save_array_atomic(feat_train_path, features_train_array)
        _save_array_atomic(feat_val_path, features_val_array)
        features_train_array = np.load(
            feat_train_path, mmap_mode='r')
        features_val_array = np.load(
            feat_val_path, mmap_mode='r')

    T_tokens = features_train_array.shape[1]
    D = features_train_array.shape[2]
    print(f"  D={D}, T={T_tokens}")

    # ---- Build result skeleton ----
    config = dict(vars(args))
    config['alpha_list'] = alpha_list
    config['alpha_weights_parsed'] = alpha_weights
    results = build_result_skeleton(config, seeds, manifest)
    results['run_id'] = args.run_id
    results['code_version'] = CODE_VERSION
    results['response_config'] = {
        'response_objective': response_objective,
        'alpha_list': alpha_list,
        'alpha_weights': alpha_weights,
        'beta': args.beta,
        'beta_target_ratio': args.beta_target_ratio,
        'd0_normalize': args.d0_normalize,
        'K': args.K,
        'max_rate_bpt': float(bits_per_token),
    }

    git_porcelain = git_status_porcelain(V1_ROOT)
    if git_porcelain is not None:
        results['git_status_porcelain'] = git_porcelain

    # ---- Load backbone ----
    print(f"\nLoading DINOv2 ({args.backbone})...")
    wrapper = Dinov2Wrapper(
        head_layers=1, model_name=args.backbone, device=device)
    n_blocks = len(wrapper.backbone.blocks)
    tail_blocks = list(wrapper.backbone.blocks[layer_idx + 1:])
    norm_layer = wrapper.backbone.norm
    print(f"  Tail: {len(tail_blocks)} blocks + norm")

    # ================================================================
    #  (A) OPQ baseline
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"  [OPQ] K={args.K} baseline")
    print(f"{'=' * 60}")

    opq_groups = D // args.embedding_dim
    opq_art_path = (
        os.path.realpath(args.opq_artifact)
        if args.opq_artifact else _opq_artifact_path_v1_1(args, seeds))

    opq_expected_meta = {
        'layer': args.layer, 'K': args.K,
        'embedding_dim': args.embedding_dim,
        'norm_mode': args.norm_mode,
        'n_train': args.n_train,
        'split_seed': seeds['split'],
        'init_seed': seeds['init'],
        'opq_iter': args.opq_iter,
        'kmeans_iter': args.kmeans_iter,
        'kmeans_max_samples': args.kmeans_max_samples,
        'train_feature_ids': manifest['train_basenames'],
    }
    if not args.opq_artifact:
        opq_expected_meta['code_version'] = CODE_VERSION

    if os.path.isfile(opq_art_path):
        print(f"  Loading OPQ artifact: {opq_art_path}")
        R_std, codebooks_std, opq_meta = load_opq_artifact(
            opq_art_path, opq_groups, expected_meta=opq_expected_meta)
        std_time = 0.0
    else:
        lock_path = _opq_lock_path(opq_art_path)
        print(f"  Acquiring OPQ lock: {lock_path}")
        _acquire_opq_lock(lock_path)
        try:
            if os.path.isfile(opq_art_path):
                print(f"  OPQ created by another process, loading...")
                R_std, codebooks_std, opq_meta = load_opq_artifact(
                    opq_art_path, opq_groups,
                    expected_meta=opq_expected_meta)
                std_time = 0.0
            else:
                wrapper.backbone.cpu()
                if wrapper.head is not None:
                    wrapper.head.cpu()
                torch.cuda.empty_cache()

                t0 = time.time()
                all_vectors = []
                for start in range(
                        0, len(features_train_array), 200):
                    end = min(start + 200,
                              len(features_train_array))
                    X = torch.from_numpy(
                        features_train_array[start:end]
                    ).float().to(device)
                    with torch.no_grad():
                        Y, _, _ = batch_normalize_gpu(
                            X, mode=args.norm_mode)
                    all_vectors.append(
                        Y.reshape(-1, D).cpu().numpy())
                    del X, Y
                full_vectors = np.concatenate(all_vectors, axis=0)
                del all_vectors

                max_flat = args.kmeans_max_samples // opq_groups
                if full_vectors.shape[0] > max_flat:
                    rng2 = np.random.RandomState(seeds['init'])
                    idx2 = rng2.choice(
                        full_vectors.shape[0], max_flat,
                        replace=False)
                    full_vectors = full_vectors[idx2]

                R_std, codebooks_std, hist_std = \
                    learn_opq_rotation(
                        full_vectors, opq_groups,
                        args.embedding_dim, args.K,
                        max_iter_opq=args.opq_iter,
                        max_iter_kmeans=args.kmeans_iter,
                        device=device, verbose=False)
                del full_vectors
                torch.cuda.empty_cache()
                std_time = time.time() - t0
                print(f"  OPQ done: MSE={hist_std[-1][0]:.8f} "
                      f"({std_time:.1f}s)")

                save_opq_artifact(
                    opq_art_path, R_std, codebooks_std,
                    hist_std, opq_expected_meta)
                print(f"  Saved OPQ artifact: {opq_art_path}")

                wrapper.backbone.to(device)
                if wrapper.head is not None:
                    wrapper.head.to(device)
                torch.cuda.empty_cache()
        finally:
            _release_opq_lock(lock_path)

    # K256 sanity: label range
    if args.K == 256:
        results['k256_sanity'] = {
            'max_rate_bpt': float(bits_per_token),
            'label_range': [0, args.K - 1],
        }
        assert bits_per_token == 256, \
            f"K256 rate must be 256 bpt, got {bits_per_token}"
        opq_min, opq_max = args.K, -1
        codebooks_tensor = torch.as_tensor(
            np.stack(codebooks_std), device=device, dtype=torch.float32)
        for start in range(0, len(features_val_array), args.batch_size):
            end = min(start + args.batch_size, len(features_val_array))
            X_label = torch.from_numpy(
                np.asarray(features_val_array[start:end])
            ).float().to(device)
            with torch.no_grad():
                Y_label, _, _ = batch_normalize_gpu(
                    X_label, mode=args.norm_mode)
                Z_label = Y_label.reshape(-1, D) @ torch.as_tensor(
                    R_std, device=device, dtype=Y_label.dtype)
                sub_label = Z_label.reshape(
                    -1, opq_groups, args.embedding_dim
                ).permute(1, 0, 2)
                _, labels_label = batched_assign(
                    sub_label, codebooks_tensor, device=device)
            opq_min = min(opq_min, int(labels_label.min().item()))
            opq_max = max(opq_max, int(labels_label.max().item()))
            del X_label, Y_label, Z_label, sub_label, labels_label
        results['k256_sanity']['opq_val_label_min'] = opq_min
        results['k256_sanity']['opq_val_label_max'] = opq_max
        results['k256_sanity']['opq_val_labels_in_range'] = bool(
            opq_min >= 0 and opq_max <= 255)
        if not results['k256_sanity']['opq_val_labels_in_range']:
            raise RuntimeError(
                f"K256 OPQ labels out of range: [{opq_min}, {opq_max}]")
        del codebooks_tensor

    tail = FrozenTail(tail_blocks, norm_layer, device=device)

    # OPQ evaluation
    if args.skip_test_eval:
        acc_std = None
        std_delta_l = evaluate_delta_l_ref(
            features_val_array, tail, args.norm_mode, device,
            codebooks=codebooks_std, R=R_std,
            embedding_dim=args.embedding_dim,
            batch_size=args.batch_size)
        results['std_opq_val_D0_raw'] = float(std_delta_l)
        results['std_opq_val_D0_per_element'] = float(
            std_delta_l / (T_tokens * D))
        print(f"  * OPQ validation D0 = {std_delta_l:.1f} "
              f"(per_elem={std_delta_l/(T_tokens*D):.4f})")
    else:
        opq_reference = None
        if args.opq_reference_json:
            opq_reference = _load_test_reference(
                args.opq_reference_json, args, manifest)
            acc_std = opq_reference['accuracy']
            results['std_opq_accuracy_reference'] = opq_reference
            print(f"  * Reusing frozen OPQ Acc = {acc_std:.4f} "
                  f"from {args.opq_reference_json}")
        else:
            xhat_std = pq_encode_decode_features(
                features_test, codebooks_std, args.embedding_dim,
                args.norm_mode, device, R=R_std)
            wrapper.backbone.to(device)
            if wrapper.head is not None:
                wrapper.head.to(device)
            torch.cuda.empty_cache()
            acc_std = evaluate_accuracy(
                xhat_std, basenames_test, gt_test,
                wrapper, layer_idx, device)
            del xhat_std
        results['std_opq_acc'] = float(acc_std)
        print(f"  * OPQ Acc = {acc_std:.4f}")
        if opq_reference is not None and opq_reference['D0'] is not None:
            std_delta_l = float(opq_reference['D0'])
            print(f"  * Reusing frozen OPQ D0 = {std_delta_l:.1f}")
        else:
            std_delta_l = evaluate_delta_l_ref(
                features_test, tail, args.norm_mode, device,
                codebooks=codebooks_std, R=R_std,
                embedding_dim=args.embedding_dim,
                batch_size=args.batch_size)
        results['std_opq_D0_raw'] = float(std_delta_l)
        results['std_opq_D0_per_element'] = float(
            std_delta_l / (T_tokens * D))
        print(f"  * OPQ D0 = {std_delta_l:.1f} "
              f"(per_elem={std_delta_l/(T_tokens*D):.4f})")

    # ---- Shared teacher cache ----
    tc_train_path = _teacher_cache_path(
        args, seeds, 'train', args.n_train)
    tc_val_path = _teacher_cache_path(
        args, seeds, 'val', args.n_val)

    def _make_train_tc():
        print(f"  Computing shared teacher cache "
              f"(train, {args.n_train} images)...")
        return _compute_teacher_cache(
            features_train_array, tail, args.batch_size, device)

    def _make_val_tc():
        print(f"  Computing shared teacher cache "
              f"(val, {args.n_val} images)...")
        return _compute_teacher_cache(
            features_val_array, tail, args.batch_size, device)

    t_tc = time.time()
    teacher_cache_train = None
    if not args.eval_checkpoint:
        teacher_cache_train = _load_or_create_array(
            tc_train_path, _make_train_tc, mmap=True)
    teacher_cache_val = _load_or_create_array(
        tc_val_path, _make_val_tc, mmap=True)
    train_cache_text = (
        f"{teacher_cache_train.nbytes/1e9:.1f}GB"
        if teacher_cache_train is not None else "skipped (eval-only)")
    print(f"  Teacher caches ready: "
          f"train={train_cache_text} "
          f"val={teacher_cache_val.nbytes/1e9:.1f}GB "
          f"({time.time()-t_tc:.1f}s)")

    # ================================================================
    #  (B) V1.1 Codec training
    # ================================================================
    if response_objective == 'none':
        print(f"\n  response_objective=none, β={args.beta}: "
              f"OPQ-only evaluation, skipping codec training.")
        results['phase'] = 'OPQ_only'
        results['train_time_std'] = float(std_time)

        tag = (f"{args.layer}_K{args.K}_opq_s{args.seed}")
        if args.result_suffix:
            tag += f"_{args.result_suffix}"
        _result_json_atomic(
            os.path.join(out_dir, f'{tag}.json'), results)
        print(f"\nResults: {os.path.join(out_dir, tag + '.json')}")
        return results

    print(f"\n{'=' * 60}")
    print(f"  [V1.1 Codec] Phase {phase_tag}  "
          f"objective={response_objective}  β={args.beta}")
    print(f"{'=' * 60}")

    for i, blk in enumerate(wrapper.backbone.blocks):
        if i <= layer_idx:
            blk.cpu()
    if wrapper.head is not None:
        wrapper.head.cpu()
    torch.cuda.empty_cache()

    fz_tag = ""
    if args.freeze_transform:
        fz_tag += "_fzR"
    if args.freeze_codebooks:
        fz_tag += "_fzC"
    step_tag = f"_{args.step_mode}"
    if args.step_mode == 'alternating':
        step_tag += f"_u{args.alt_u_steps}c{args.alt_c_steps}"
    weight_tag = "w" + "-".join(f"{w:g}" for w in alpha_weights)
    ratio_tag = (
        f"_gr{args.beta_target_ratio:g}"
        if args.beta_target_ratio > 0 else "")

    if args.eval_checkpoint:
        ckpt_path = os.path.realpath(args.eval_checkpoint)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        codec = load_codec_v1(ckpt_path, device=device)
        if (codec.pq.G != num_groups or codec.pq.K != args.K
                or codec.pq.d != args.embedding_dim):
            raise ValueError(
                "Checkpoint architecture mismatch: "
                f"got G={codec.pq.G},K={codec.pq.K},d={codec.pq.d}; "
                f"expected G={num_groups},K={args.K},"
                f"d={args.embedding_dim}")
        transform = codec.transform
        history = []
        train_time = 0.0
        results['evaluation_only'] = True
        results['checkpoint_source'] = ckpt_path
        print(f"  Loaded frozen checkpoint: {ckpt_path}")
    else:
        # Build transform and warm-start from the frozen OPQ artifact.
        transform = OrthogonalTransform(D) if bt_dim == D else None
        if bt_dim > 0 and bt_dim != D:
            transform = FeatureTransform(D, bt_dim)

        R_ws = R_std.copy()
        C_ws = [c.copy() for c in codebooks_std]
        if np.linalg.det(R_ws) < 0:
            R_ws[:, -1] *= -1
            C_ws[-1][:, -1] *= -1
            print(f"  det(R_opq)<0: flipped to SO(D)")

        if transform is not None and hasattr(transform, 'init_from_opq'):
            transform_check = OrthogonalTransform(D).to(device)
            transform_check.init_from_opq(R_ws)
            cayley_check = check_cayley_init(
                transform_check, R_ws, verbose=True)
            results['cayley_init_check'] = cayley_check
            if cayley_check['rel_error'] > 0.1:
                raise RuntimeError(
                    f"Cayley init failed: "
                    f"rel_error={cayley_check['rel_error']:.4f}")
            del transform_check

        t0 = time.time()
        codec, history = train_v1(
            features_array=features_train_array,
            T_tokens=T_tokens,
            tail=tail,
            G=num_groups, K=args.K, d=args.embedding_dim,
            norm_mode=args.norm_mode,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            device=device,
            seeds=seeds,
            val_array=features_val_array,
            verbose=True,
            transform=transform,
            R_init=R_ws,
            codebooks_init=C_ws,
            kmeans_max_samples=args.kmeans_max_samples,
            beta=args.beta,
            alpha=alpha_list[0],
            elastic_tau=args.elastic_tau,
            n_groups_per_batch=args.n_groups_per_batch,
            compute_diagnostics=True,
            n_interaction_pairs=args.n_interaction_pairs,
            tau_start=args.tau_start,
            tau_end=args.tau_end,
            tau_schedule=args.tau_schedule,
            freeze_transform=args.freeze_transform,
            freeze_codebooks=args.freeze_codebooks,
            lmbda=0.0,
            grad_clip=args.grad_clip,
            step_mode=args.step_mode,
            alt_u_steps=args.alt_u_steps,
            alt_c_steps=args.alt_c_steps,
            val_elasticity_interval=args.val_elasticity_interval,
            probe_group_chunk=args.probe_group_chunk,
            heldout_probe_group_chunk=args.heldout_probe_group_chunk,
            teacher_cache_precomputed=teacher_cache_train,
            val_teacher_cache_precomputed=teacher_cache_val,
            response_objective=response_objective,
            alpha_list=alpha_list,
            alpha_weights=alpha_weights,
            d0_normalize=args.d0_normalize,
            beta_target_ratio=args.beta_target_ratio,
        )
        train_time = time.time() - t0
        print(f"  Training: {train_time:.1f}s")

        ckpt_dir = os.path.join(
            V1_ROOT, 'checkpoints', args.backbone, args.run_id)
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_name = (f"{args.layer}_K{args.K}_emb{args.embedding_dim}"
                     f"_{response_objective}_{weight_tag}"
                     f"_b{args.beta}{ratio_tag}_a{args.alpha_list}"
                     f"_lr{args.lr}_gpb{args.n_groups_per_batch}"
                     f"{fz_tag}{step_tag}"
                     f"_ep{args.epochs}_s{args.seed}")
        if args.result_suffix:
            ckpt_name += f"_{args.result_suffix}"
        ckpt_path = os.path.join(ckpt_dir, f"{ckpt_name}.pt")
        save_codec_v1(codec, ckpt_path)
        print(f"  Saved: {ckpt_path}")

    results['history'] = history
    results['checkpoint'] = ckpt_path
    if history:
        results['response_config']['effective_beta'] = history[-1].get(
            'effective_beta', args.beta)
        results['response_config']['beta_calibration'] = history[-1].get(
            'beta_calibration')

    # Reconstruction audit
    print(f"\n  Reconstruction audit...")
    audit_features = (
        features_val_array[:50] if args.skip_test_eval
        else features_test[:50])
    audit_info = reconstruction_audit(
        audit_features, codec, args.norm_mode, device,
        batch_size=args.batch_size, return_details=True)
    results['reconstruction_audit'] = audit_info
    print(f"  Audit: quant={audit_info['quantisation_relative_error']:.2e}"
          f" combined={audit_info['combined_relative_error']:.2e}"
          f" {'PASS' if audit_info['passed'] else 'FAIL'}")
    if not audit_info['passed']:
        raise RuntimeError(
            f"Reconstruction audit failed: {audit_info}")

    if transform and hasattr(transform, 'orth_error'):
        results['final_orth_error'] = float(transform.orth_error())

    # ---- Comprehensive validation held-out (§15.7) ----
    print(f"\n  Computing comprehensive validation held-out metrics...")
    val_el = evaluate_heldout_v1_1(
        features_val_array, teacher_cache_val, codec, tail,
        num_groups, args.norm_mode, args.batch_size, device,
        alphas=alpha_list,
        probe_group_chunk=args.heldout_probe_group_chunk)
    results.setdefault('heldout_metrics', {})['val'] = val_el
    a_main = alpha_list[-1]
    m_key = f'M_g_alpha{a_main}'
    s_key = f'S_g_alpha{a_main}'
    if m_key in val_el:
        m_s = val_el[m_key]
        print(f"  Val M_g(α={a_main}): "
              f"mean={m_s['global_mean']:.4f} "
              f"span={m_s['global_span']:.4f} "
              f"pairwise={m_s['pairwise_dispersion']:.6f}")
    if s_key in val_el:
        s_s = val_el[s_key]
        print(f"  Val S_g(α={a_main}): "
              f"mean={s_s['global_mean']:.4f} "
              f"span={s_s['global_span']:.4f}")

    if args.K == 256:
        val_labels = _codec_labels(
            [features_val_array[i]
             for i in range(features_val_array.shape[0])],
            codec, args.norm_mode, device,
            batch_size=args.batch_size)
        labels_ok = bool(
            val_labels.min() >= 0 and val_labels.max() <= 255)
        results['k256_sanity']['val_label_min'] = int(val_labels.min())
        results['k256_sanity']['val_label_max'] = int(val_labels.max())
        results['k256_sanity']['val_labels_in_range'] = labels_ok
        if not labels_ok:
            raise RuntimeError(
                f"K256 validation labels out of range: "
                f"[{val_labels.min()}, {val_labels.max()}]")
        del val_labels

    if args.skip_test_eval:
        results['train_time_std'] = float(std_time)
        results['train_time_v1_1'] = float(train_time)
        results['phase'] = phase_tag

        tag = (f"{args.layer}_K{args.K}_{response_objective}"
               f"_{weight_tag}_b{args.beta}{ratio_tag}_lr{args.lr}"
               f"_gpb{args.n_groups_per_batch}"
               f"{fz_tag}{step_tag}"
               f"_ep{args.epochs}_s{args.seed}")
        if args.result_suffix:
            tag += f"_{args.result_suffix}"
        result_path = os.path.join(out_dir, f'{tag}.json')
        _result_json_atomic(result_path, results)
        print(f"\nValidation-only results: {result_path}")
        return results

    # ---- Held-out on test set ----
    if _use_elastic:
        print(f"\n  Computing test-set held-out metrics...")
        n_test = len(features_test)
        tc_test_path = _teacher_cache_path(
            args, seeds, 'test', n_test)
        feat_test_path = _features_cache_path(
            args, seeds, 'test', n_test)

        test_array = _load_or_create_array(
            feat_test_path,
            lambda: np.stack(features_test), mmap=True)
        test_teacher_cache = _load_or_create_array(
            tc_test_path,
            lambda: _compute_teacher_cache(
                test_array, tail, args.batch_size, device),
            mmap=True)

        test_el = evaluate_heldout_v1_1(
            test_array, test_teacher_cache, codec, tail,
            num_groups, args.norm_mode, args.batch_size, device,
            alphas=alpha_list,
            probe_group_chunk=args.heldout_probe_group_chunk)
        results.setdefault('heldout_metrics', {})['test'] = test_el
        if m_key in test_el:
            m_t = test_el[m_key]
            print(f"  Test M_g(α={a_main}): "
                  f"mean={m_t['global_mean']:.4f} "
                  f"span={m_t['global_span']:.4f}")
        del test_array, test_teacher_cache

    tail.to('cpu')
    torch.cuda.empty_cache()

    # ================================================================
    #  (C) Test-set evaluation
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"  [Evaluation] test set")
    print(f"{'=' * 60}")

    codec_reference = None
    if args.codec_reference_json:
        if not args.eval_checkpoint:
            raise ValueError(
                "codec_reference_json requires eval_checkpoint so the "
                "referenced accuracy is tied to an exact frozen checkpoint")
        codec_reference = _load_test_reference(
            args.codec_reference_json, args, manifest,
            expected_checkpoint=ckpt_path)
        acc_v1 = codec_reference['accuracy']
        results['v1_1_accuracy_reference'] = codec_reference
        print(f"  * Reusing frozen codec Acc = {acc_v1:.4f} "
              f"from {args.codec_reference_json}")
    else:
        xhat_v1 = codec_encode_decode(
            features_test, codec, args.norm_mode, device)

        wrapper.backbone.to(device)
        if wrapper.head is not None:
            wrapper.head.to(device)
        torch.cuda.empty_cache()

        acc_v1 = evaluate_accuracy(
            xhat_v1, basenames_test, gt_test,
            wrapper, layer_idx, device)
        del xhat_v1
    results['v1_1_acc'] = float(acc_v1)
    delta_acc = acc_v1 - (acc_std or 0)
    print(f"  * V1.1 Acc = {acc_v1:.4f} (Δ={delta_acc:+.4f})")

    tail_blocks2 = list(wrapper.backbone.blocks[layer_idx + 1:])
    norm2 = wrapper.backbone.norm
    tail2 = FrozenTail(tail_blocks2, norm2, device=device)

    v1_delta_l = evaluate_delta_l_ref(
        features_test, tail2, args.norm_mode, device,
        codec=codec, batch_size=args.batch_size)
    results['v1_1_D0_raw'] = float(v1_delta_l)
    results['v1_1_D0_per_element'] = float(
        v1_delta_l / (T_tokens * D))
    delta_dl = v1_delta_l - std_delta_l
    print(f"  * V1.1 D0 = {v1_delta_l:.1f} "
          f"(per_elem={v1_delta_l/(T_tokens*D):.4f}, "
          f"Δ={delta_dl:+.1f})")

    tail2.to('cpu')
    torch.cuda.empty_cache()

    # Rate
    print(f"  Computing rate metrics...")
    train_labels = _codec_labels(
        [features_train_array[i]
         for i in range(features_train_array.shape[0])],
        codec, args.norm_mode, device,
        batch_size=args.batch_size)
    train_pmf = _histogram_pmf(train_labels, num_groups, args.K)
    rate_info = evaluate_rate(
        features_test, codec, args.norm_mode, device,
        batch_size=args.batch_size, train_pmf=train_pmf)
    results['rate_info'] = rate_info

    # K256 sanity checks
    if args.K == 256:
        results['k256_sanity']['labels_in_range'] = bool(
            train_labels.min() >= 0 and train_labels.max() <= 255)
        results['k256_sanity']['empirical_entropy'] = float(
            rate_info['empirical_entropy_bpt'])
        print(f"  K256 sanity: labels [{train_labels.min()}, "
              f"{train_labels.max()}], "
              f"H={rate_info['empirical_entropy_bpt']:.2f} bpt")

    print(f"  * H_emp={rate_info['empirical_entropy_bpt']:.2f} "
          f"max={rate_info['max_rate_bpt']:.0f} bits/token")
    if 'rans_train_bpt' in rate_info:
        print(f"  * rANS_train={rate_info['rans_train_bpt']:.2f} "
              f"bits/token")

    # ================================================================
    #  Summary
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"  Summary: {args.layer} K={args.K} "
          f"{response_objective} β={args.beta} Phase {phase_tag}")
    print(f"  OPQ:  Acc={acc_std or 0:.4f}  D0={std_delta_l:.1f}")
    print(f"  V1.1: Acc={acc_v1:.4f}  D0={v1_delta_l:.1f}")
    print(f"  Δ(Acc)={delta_acc:+.4f}  Δ(D0)={delta_dl:+.1f}")
    print(f"  bits/token: {bits_per_token:.0f}")
    print(f"{'=' * 60}")

    results['delta_acc'] = float(delta_acc)
    results['delta_dl'] = float(delta_dl)
    results['train_time_std'] = float(std_time)
    results['train_time_v1_1'] = float(train_time)
    results['phase'] = phase_tag

    tag = (f"{args.layer}_K{args.K}_{response_objective}"
           f"_{weight_tag}_b{args.beta}{ratio_tag}_lr{args.lr}"
           f"_gpb{args.n_groups_per_batch}"
           f"{fz_tag}{step_tag}"
           f"_ep{args.epochs}_s{args.seed}")
    if args.result_suffix:
        tag += f"_{args.result_suffix}"
    result_path = os.path.join(out_dir, f'{tag}.json')
    _result_json_atomic(result_path, results)
    print(f"\nResults: {result_path}")
    return results


# ================================================================
#  CLI
# ================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ORFC-v1.1 elastic experiment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Architecture
    parser.add_argument("--layer", default="blk20")
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--bottleneck_dim", type=int, default=1024)
    parser.add_argument("--norm_mode", default="per_image")

    # V1.1: response objective (§15.4)
    parser.add_argument(
        "--response_objective", default="fixed_energy",
        choices=["legacy", "fixed_energy", "operational", "none"],
        help="Response loss formulation")
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument(
        "--beta_target_ratio", type=float, default=0.0,
        help="If >0, calibrate beta on the first minibatch so the "
             "response/D0 gradient-norm ratio equals this value")
    parser.add_argument("--alpha_list", default="0.1,0.5,1.0",
                        help="Comma-separated alpha list")
    parser.add_argument("--alpha_weights", default="",
                        help="Comma-separated alpha weights "
                             "(empty = equal)")
    parser.add_argument("--d0_normalize", default="per_element",
                        choices=["raw", "per_element"],
                        help="D0 normalisation for optimiser")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--elastic_tau", type=float, default=1.0)
    parser.add_argument("--n_groups_per_batch", type=int, default=4)
    parser.add_argument("--n_interaction_pairs", type=int, default=4)

    # Temperature
    parser.add_argument("--tau_start", type=float, default=0.5)
    parser.add_argument("--tau_end", type=float, default=0.005)
    parser.add_argument("--tau_schedule", default="exponential")

    # Freezing
    parser.add_argument("--freeze_transform", action="store_true")
    parser.add_argument("--freeze_codebooks", action="store_true")

    # Phase C
    parser.add_argument("--step_mode", default="joint",
                        choices=["joint", "alternating"])
    parser.add_argument("--alt_u_steps", type=int, default=1)
    parser.add_argument("--alt_c_steps", type=int, default=1)

    # Held-out
    parser.add_argument("--val_elasticity_interval", type=int,
                        default=20)
    parser.add_argument("--probe_group_chunk", type=int, default=4)
    parser.add_argument("--heldout_probe_group_chunk", type=int,
                        default=4)

    # Data
    parser.add_argument("--n_train", type=int, default=4500)
    parser.add_argument("--n_val", type=int, default=500)
    parser.add_argument("--kmeans_max_samples", type=int,
                        default=2_000_000)
    parser.add_argument("--opq_iter", type=int, default=20)
    parser.add_argument("--kmeans_iter", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_test", type=int, default=0)
    parser.add_argument("--skip_test_eval", action="store_true")
    parser.add_argument(
        "--eval_checkpoint", default="",
        help="Load this exact frozen codec checkpoint and skip training")
    parser.add_argument(
        "--opq_artifact", default="",
        help="Explicit frozen OPQ artifact; structural metadata is validated")
    parser.add_argument(
        "--opq_reference_json", default="",
        help="Reuse OPQ test accuracy from a contract-matched frozen result")
    parser.add_argument(
        "--codec_reference_json", default="",
        help="Reuse codec test accuracy for eval_checkpoint after validating "
             "checkpoint and split identity")

    parser.add_argument("--feat_root",
                        default=os.path.join(PROJECT_ROOT, "features"))
    parser.add_argument("--train_subset", default="train")
    parser.add_argument("--test_subset", default="test")
    parser.add_argument("--backbone", default="dinov2_vitl14")
    parser.add_argument("--gt_path",
                        default=os.path.join(
                            PROJECT_ROOT, "utils",
                            "imagenet_selected_label500.txt"))

    parser.add_argument("--result_suffix", default="")
    parser.add_argument("--run_id", default="v1_1_manual")

    args = parser.parse_args()
    run_experiment(args)


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""Fast ORFC-v1 unit and contract tests.

This suite uses a tiny synthetic codec/tail, so it checks the measurement and
serialization contracts without loading DINOv2 or ImageNet features.
"""

import json
import os
import tempfile

import numpy as np
import torch
import torch.nn as nn

from codec_v1 import (
    FeatureCodecV1,
    reconstruction_audit,
    save_codec_v1,
    load_codec_v1,
)
from elastic import (
    compute_elasticity, elastic_loss, compute_loss_v1,
    pairwise_dispersion_loss, energy_match_residuals,
    compute_single_sided_probe,
)
from train_v1 import evaluate_heldout_elasticity, evaluate_heldout_v1_1

import sys
_ORFC_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', 'orfc'))
if _ORFC_DIR not in sys.path:
    sys.path.insert(0, _ORFC_DIR)
from opq import batch_normalize_gpu, batch_inv_normalize_gpu
from soft_pq import SoftPQ, OrthogonalTransform


class TinyTail(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x + 0.07 * x.square() + 0.01 * x.roll(1, dims=-1)

    @torch.no_grad()
    def forward_nograd(self, x):
        return self.forward(x)


def _assert_close(a, b, atol=1e-6, rtol=1e-5, label='value'):
    if not torch.allclose(
            torch.as_tensor(a), torch.as_tensor(b), atol=atol, rtol=rtol):
        raise AssertionError(f'{label} mismatch: {a!r} vs {b!r}')


def _make_codec(device='cpu', K=8):
    torch.manual_seed(11)
    pq = SoftPQ(G=4, K=K, d=2).to(device)
    transform = OrthogonalTransform(8).to(device)
    codec = FeatureCodecV1(pq, transform).to(device)
    with torch.no_grad():
        pq.codebooks.copy_(torch.randn_like(pq.codebooks) * 0.4)
    pq.temperature = 0.0
    return codec


def test_detail_contracts(device):
    codec = _make_codec(device)
    y = torch.randn(2, 3, 8, device=device)

    yhat, train_info = codec(
        y, return_details=True, differentiable_groups=[0, 2],
        detail_level='train')
    forbidden = {'e_g', 'Z', 'Z_hat', 'labels'}
    assert not (forbidden & set(train_info))
    assert set(train_info['e_g_diff']) == {0, 2}
    assert train_info['r_g'].shape == (4, 6, 2)
    assert train_info['r_g'].dtype == y.dtype
    assert train_info['r_g'].device == y.device
    assert all(v.requires_grad for v in train_info['e_g_diff'].values())

    yhat_diag, diag = codec(
        y, return_details=True, detail_level='diagnostic')
    assert {'e_g', 'Z', 'Z_hat', 'labels'} <= set(diag)
    assert diag['e_g'].shape == (4, 6, 8)
    rotation = codec.transform.get_rotation()
    y_roundtrip = (y.reshape(-1, 8) @ rotation @ rotation.t()).reshape_as(y)
    decomposition = diag['e_g'].sum(0).reshape_as(y)
    _assert_close(
        yhat_diag - y_roundtrip, decomposition,
        atol=2e-6, rtol=2e-5, label='diagnostic decomposition')
    _assert_close(yhat, yhat_diag, label='detail-level forward')


def _synthetic_probe(chunk, device):
    torch.manual_seed(19)
    u = nn.Parameter(torch.tensor(0.13, device=device))
    c = nn.Parameter(torch.tensor(-0.21, device=device))
    base = torch.randn(2, 3, 8, device=device)
    u_axis = torch.randn_like(base) * 0.1
    c_axis = torch.randn_like(base) * 0.1
    xhat = base + u * u_axis + c * c_axis
    std = torch.ones(2, 1, 1, device=device)
    tail = TinyTail().to(device)
    teacher = tail.forward_nograd(base * 0.8).detach()
    d0 = ((teacher - tail(xhat)) ** 2).reshape(2, -1).sum(1).mean()
    residuals = {}
    for g in range(4):
        axis = torch.randn_like(base) * (0.04 + 0.01 * g)
        residuals[g] = (
            axis + (g + 1) * 0.03 * u * u_axis
            + (4 - g) * 0.02 * c * c_axis
        ).reshape(-1, 8)
    eps, _, _ = compute_elasticity(
        xhat, residuals, std, tail, teacher, d0, alpha=0.1,
        probe_group_chunk=chunk)
    elastic = elastic_loss(eps, tau=0.05)
    assert elastic.requires_grad
    loss = compute_loss_v1(d0, elastic, beta=0.03)
    loss.backward()
    grads = torch.stack([u.grad.detach(), c.grad.detach()])
    assert torch.isfinite(grads).all()
    assert grads.abs().min() > 0
    return (
        torch.stack([eps[g].detach() for g in range(4)]),
        elastic.detach(), grads, tail.calls,
    )


def test_probe_chunk_and_beta(device):
    eps1, loss1, grads1, calls1 = _synthetic_probe(1, device)
    eps4, loss4, grads4, calls4 = _synthetic_probe(4, device)
    _assert_close(eps1, eps4, atol=1e-6, rtol=1e-6, label='chunk epsilon')
    _assert_close(loss1, loss4, atol=1e-6, rtol=1e-6, label='chunk loss')
    relative = (grads1 - grads4).norm() / grads1.norm().clamp_min(1e-12)
    assert relative.item() <= 1e-5
    assert calls1 - calls4 == 3

    # A one-step beta=0 update and beta>0 update must diverge from the same
    # initial state/minibatch.  Their difference is the elastic gradient.
    torch.manual_seed(23)
    initial = torch.tensor([0.12, -0.18], device=device)
    axes = torch.randn(2, 2, 3, 8, device=device) * 0.1
    base = torch.randn(2, 3, 8, device=device)
    teacher = TinyTail().to(device).forward_nograd(base * 0.75).detach()

    def one_step(beta):
        p = nn.Parameter(initial.clone())
        opt = torch.optim.SGD([p], lr=0.02)
        tail = TinyTail().to(device)
        xhat = base + p[0] * axes[0] + p[1] * axes[1]
        d0 = ((teacher - tail(xhat)) ** 2).reshape(2, -1).sum(1).mean()
        residuals = {
            g: ((0.03 + g * 0.01) * axes[g % 2]
                + (g + 1) * 0.01 * p[0] * axes[1]).reshape(-1, 8)
            for g in range(4)
        }
        eps, _, _ = compute_elasticity(
            xhat, residuals, torch.ones(2, 1, 1, device=device),
            tail, teacher, d0, 0.1, probe_group_chunk=4)
        objective = compute_loss_v1(d0, elastic_loss(eps, 0.05), beta)
        opt.zero_grad()
        objective.backward()
        opt.step()
        return p.detach()

    assert not torch.equal(one_step(0.0), one_step(0.03))


def _metric_vector(result):
    values = []
    for key in ('eps_g', 'q_g_normalized', 'q_g_original', 'kappa_g'):
        for g in range(4):
            values.append(result[key]['per_group'][g]['mean'])
    return np.asarray(values, dtype=np.float64)


def test_heldout_and_audit(device):
    rng = np.random.RandomState(29)
    features = rng.randn(7, 3, 8).astype(np.float32)
    codec = _make_codec(device)
    tail = TinyTail().to(device)
    with torch.no_grad():
        teacher = tail(
            torch.from_numpy(features).to(device)).cpu().numpy()

    def evaluate(x, y, batch):
        tail.calls = 0
        result = evaluate_heldout_elasticity(
            x, y, codec, tail, G=4, alpha=0.1,
            norm_mode='per_image', batch_size=batch, device=device,
            probe_group_chunk=2)
        expected_batches = int(np.ceil(len(x) / batch))
        expected_calls = expected_batches * (1 + 2 + 1)
        assert tail.calls == expected_calls
        return result

    r2 = evaluate(features, teacher, 2)
    r3 = evaluate(features, teacher, 3)
    perm = np.array([6, 1, 4, 0, 5, 2, 3])
    rp = evaluate(features[perm], teacher[perm], 3)
    base = _metric_vector(r2)
    for label, other in [('batch', r3), ('permutation', rp)]:
        vec = _metric_vector(other)
        relative = np.max(np.abs(vec - base) / np.maximum(np.abs(base), 1e-8))
        assert relative <= 1e-4, f'{label} relative error {relative}'

    for g in range(4):
        assert r2['eps_g']['per_group'][g]['n'] == 7
        assert r2['q_g_normalized']['per_group'][g]['n'] == 7
        assert r2['q_g_original']['per_group'][g]['n'] == 7
        assert r2['kappa_g']['per_group'][g]['n'] == 7
    assert r2['interaction']['per_pair']
    assert all(row['n'] > 0
               for row in r2['interaction']['per_pair'].values())
    assert not r2['performance_contract']['group_by_image_tail_loop']

    audit = reconstruction_audit(
        features, codec, 'per_image', device, batch_size=3,
        return_details=True)
    assert audit['passed'], json.dumps(audit, indent=2)
    assert audit['quantisation_relative_error'] <= 1e-5
    assert 'roundtrip_relative_magnitude' in audit


def test_checkpoint_and_opq_contract(device):
    codec = _make_codec(device)
    x = torch.randn(2, 3, 8, device=device)
    with tempfile.TemporaryDirectory() as td:
        checkpoint = os.path.join(td, 'codec.pt')
        save_codec_v1(codec, checkpoint)
        assert os.path.isfile(checkpoint)
        assert not any(name.endswith('.tmp') for name in os.listdir(td))
        loaded = load_codec_v1(checkpoint, device=device)
        with torch.no_grad():
            expected, _ = codec(x)
            actual, _ = loaded(x)
        _assert_close(expected, actual, label='checkpoint reload')

        # Importing the runner here keeps the core tests independent of its
        # heavier application imports until this metadata contract is checked.
        from run_v1 import save_opq_artifact, load_opq_artifact
        artifact = os.path.join(td, 'opq.npz')
        rotation = np.eye(8, dtype=np.float32)
        codebooks = [
            np.zeros((8, 2), dtype=np.float32) for _ in range(4)]
        complete = {
            'layer': 'blk20',
            'K': 8,
            'embedding_dim': 2,
            'norm_mode': 'per_image',
            'n_train': 7,
            'split_seed': 101,
            'init_seed': 202,
            'opq_iter': 3,
            'kmeans_iter': 5,
            'kmeans_max_samples': 99,
            'train_feature_ids': ['a', 'b', 'c'],
        }
        save_opq_artifact(artifact, rotation, codebooks, [], complete)
        load_opq_artifact(artifact, 4, expected_meta=complete)
        incomplete = dict(complete)
        incomplete.pop('opq_iter')
        broken = os.path.join(td, 'broken.npz')
        save_opq_artifact(broken, rotation, codebooks, [], incomplete)
        try:
            load_opq_artifact(broken, 4, expected_meta=complete)
        except ValueError:
            pass
        else:
            raise AssertionError('missing OPQ metadata was accepted')


def test_v1_1_response_contracts(device):
    # Pairwise loss: equality, permutation invariance, unbiased subset mean.
    z = torch.tensor([0.2, -0.3, 0.7, 1.1], device=device)
    assert pairwise_dispersion_loss(torch.ones_like(z)).item() == 0.0
    expected = pairwise_dispersion_loss(z)
    perm = torch.tensor([2, 0, 3, 1], device=device)
    _assert_close(
        pairwise_dispersion_loss(z[perm]), expected,
        label='pairwise permutation')
    subset_losses = []
    for i in range(len(z)):
        for j in range(i + 1, len(z)):
            subset_losses.append(pairwise_dispersion_loss(z[[i, j]]))
    _assert_close(
        torch.stack(subset_losses).mean(), expected,
        label='pairwise unbiased subsets')

    # Full-G original-space q_bar is reused by every sampled chunk and all
    # normalisation statistics are stop-gradient.
    torch.manual_seed(101)
    B, T, D = 2, 3, 8
    std = torch.tensor([0.5, 1.7], device=device).reshape(B, 1, 1)
    residuals = {
        g: nn.Parameter(
            torch.randn(B * T, D, device=device) * (0.02 + 0.03 * g))
        for g in range(4)
    }
    q_all = {}
    for g, residual in residuals.items():
        original = residual.detach().reshape(B, T, D) * std
        q_all[g] = original.square().sum((1, 2)) / T
    q_bar = torch.stack([q_all[g] for g in range(4)]).mean(0)

    matched_all = {}
    for chunk in ([0], [1, 2], [3]):
        matched, q_local, q_used, audit = energy_match_residuals(
            {g: residuals[g] for g in chunk}, B, T,
            q_bar_per_image=q_bar, Std=std)
        assert audit <= 1e-5
        assert not q_used.requires_grad
        assert all(not q.requires_grad for q in q_local.values())
        matched_all.update(matched)
    match_loss = sum(value.square().mean() for value in matched_all.values())
    match_loss.backward()
    assert all(r.grad is not None and r.grad.abs().sum() > 0
               for r in residuals.values())

    # Batched single-sided probes are chunk invariant; alpha=0 is the base.
    base = torch.randn(B, T, D, device=device)
    xhat = base + 0.1 * torch.randn_like(base)
    tail = TinyTail().to(device)
    teacher = tail.forward_nograd(base).detach()
    e_probe = {g: matched_all[g].detach() for g in range(4)}
    d1, _ = compute_single_sided_probe(
        xhat, e_probe, std, tail, teacher,
        [0.0, 0.1, 1.0], probe_chunk=1)
    d12, _ = compute_single_sided_probe(
        xhat, e_probe, std, tail, teacher,
        [0.0, 0.1, 1.0], probe_chunk=12)
    d0 = ((teacher - tail(xhat)) ** 2).reshape(B, -1).sum(1).mean()
    for key in d1:
        _assert_close(d1[key], d12[key], label=f'probe chunk {key}')
    for g in range(4):
        _assert_close(d1[(0.0, g)], d0, label=f'alpha zero group {g}')

    # alpha=1 repair agrees between rotated and original representations.
    codec = _make_codec(device)
    y = torch.randn(B, T, D, device=device)
    yhat, info = codec(
        y, return_details=True, detail_level='diagnostic')
    rotation = codec.transform.get_rotation()
    g = 2
    d = codec.pq.d
    e_orig = info['e_g'][g].reshape(B, T, D)
    repaired_original = yhat - e_orig
    z_repaired = info['Z_hat'].clone()
    z_repaired[:, g*d:(g+1)*d] -= info['r_g'][g]
    repaired_rotated = (z_repaired @ rotation.t()).reshape(B, T, D)
    _assert_close(
        repaired_original, repaired_rotated,
        atol=2e-6, rtol=2e-5, label='alpha one repair')


def _v1_1_metric_vector(result):
    values = []
    for key in (
            'S_g_alpha0.1', 'S_g_alpha1.0',
            'M_g_alpha0.1', 'M_g_alpha1.0',
            'q_g_normalized', 'q_g_original',
            'N_g_stats', 'N_fixed_g_stats'):
        for g in range(4):
            values.append(result[key]['per_group'][g]['mean'])
    values.extend([
        result['interaction']['mean'],
        result['interaction']['max_abs'],
        result['D0_raw_mean'],
    ])
    return np.asarray(values, dtype=np.float64)


def test_v1_1_heldout_invariance_and_k256(device):
    rng = np.random.RandomState(109)
    features = rng.randn(7, 3, 8).astype(np.float32)
    codec = _make_codec(device)
    tail = TinyTail().to(device)
    with torch.no_grad():
        teacher = tail(
            torch.from_numpy(features).to(device)).cpu().numpy()

    def evaluate(x, y, batch, chunk):
        return evaluate_heldout_v1_1(
            x, y, codec, tail, G=4, norm_mode='per_image',
            batch_size=batch, device=device,
            alphas=[0.1, 0.5, 1.0], probe_group_chunk=chunk)

    reference = evaluate(features, teacher, 2, 1)
    alternatives = [
        ('chunk', evaluate(features, teacher, 2, 4)),
        ('batch', evaluate(features, teacher, 3, 2)),
    ]
    permutation = np.array([6, 1, 4, 0, 5, 2, 3])
    alternatives.append((
        'order',
        evaluate(features[permutation], teacher[permutation], 3, 4)))
    base = _v1_1_metric_vector(reference)
    for label, result in alternatives:
        values = _v1_1_metric_vector(result)
        relative = np.max(
            np.abs(values - base) / np.maximum(np.abs(base), 1e-7))
        assert relative <= 2e-4, f'{label} relative error {relative}'
        assert result['energy_match_audit_max'] <= 1e-5

    # K256 checkpoint and label range contract on a compact synthetic codec.
    codec256 = _make_codec(device, K=256)
    _, info = codec256(
        torch.randn(2, 3, 8, device=device),
        return_details=True, detail_level='diagnostic')
    assert info['labels'].min().item() >= 0
    assert info['labels'].max().item() <= 255
    with tempfile.TemporaryDirectory() as td:
        checkpoint = os.path.join(td, 'tiny_K256.pt')
        save_codec_v1(codec256, checkpoint)
        loaded = load_codec_v1(checkpoint, device=device)
        assert loaded.pq.K == 256


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_detail_contracts(device)
    test_probe_chunk_and_beta(device)
    test_heldout_and_audit(device)
    test_checkpoint_and_opq_contract(device)
    test_v1_1_response_contracts(device)
    test_v1_1_heldout_invariance_and_k256(device)
    print(json.dumps({
        'status': 'PASS',
        'device': str(device),
        'tests': [
            'detail contracts',
            'probe chunk/gradient/beta step',
            'heldout batch/order/count/tail-call contracts',
            'reconstruction audit',
            'checkpoint and OPQ metadata',
            'V1.1 response/energy/alpha/chunk contracts',
            'V1.1 heldout batch/order invariance and K256',
        ],
    }, indent=2))


if __name__ == '__main__':
    main()

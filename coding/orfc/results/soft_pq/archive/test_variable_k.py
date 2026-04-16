#!/usr/bin/env python
"""Smoke test for Variable K_g implementation.

Tests gradient flow, allocate_K, split_merge_resize, and evaluation
helpers with both uniform and variable K configurations.
"""
import sys, os, math
import torch
import numpy as np

ORFC_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
if ORFC_ROOT not in sys.path:
    sys.path.insert(0, ORFC_ROOT)

from soft_pq import SoftPQ, allocate_K

G, K, d = 4, 16, 8
N = 64

def test_uniform_K_forward_backward():
    """Uniform K: forward + backward, codebook gradients flow."""
    pq = SoftPQ(G, K, d, lmbda=1.0)
    Z = torch.randn(N, G * d, requires_grad=False)
    Z_hat, usage = pq._quantise(Z)
    loss = Z_hat.pow(2).sum()
    loss.backward()
    assert pq.codebooks.grad is not None, "codebook grad is None"
    assert pq.codebooks.grad.abs().sum() > 0, "codebook grad is zero"
    # log_prior grad requires rate in loss; here we only test codebook path
    assert pq.valid_mask.all(), "valid_mask not all True for uniform K"
    assert pq.K_per_group_t.tolist() == [K] * G
    print("  [PASS] uniform K forward+backward")

def test_variable_K_forward_backward():
    """Variable K: forward + backward with different K per group."""
    K_list = [4, 8, 16, 32]
    pq = SoftPQ(G, K_list, d, lmbda=1.0)
    assert pq.K_max == 32
    assert pq.K_per_group_t.tolist() == K_list
    assert pq.codebooks.shape == (G, 32, d)
    assert pq.valid_mask.shape == (G, 32)
    for g, k in enumerate(K_list):
        assert pq.valid_mask[g, :k].all()
        assert not pq.valid_mask[g, k:].any()

    Z = torch.randn(N, G * d, requires_grad=False)
    Z_hat, usage = pq._quantise(Z)
    loss = Z_hat.pow(2).sum()
    loss.backward()
    assert pq.codebooks.grad is not None, "codebook grad is None"
    for g, k in enumerate(K_list):
        if k < 32:
            assert pq.codebooks.grad[g, k:].abs().sum() == 0, \
                f"group {g}: padded codebook entries have nonzero grad"
    labels = pq._last_labels
    for g, k in enumerate(K_list):
        assert labels[g].max() < k, \
            f"group {g}: label {labels[g].max()} >= K_g={k}"
    print("  [PASS] variable K forward+backward")

def test_variable_K_ste():
    """Variable K + STE: gradient flows to input."""
    K_list = [8, 16, 32, 64]
    pq = SoftPQ(G, K_list, d, lmbda=1.0, use_ste=True)
    Z = torch.randn(N, G * d, requires_grad=True)
    Z_hat, usage = pq._quantise(Z)
    loss = Z_hat.pow(2).sum()
    loss.backward()
    assert Z.grad is not None, "STE: input grad is None"
    assert Z.grad.abs().sum() > 0, "STE: input grad is zero"
    assert pq.codebooks.grad is not None, "STE: codebook grad is None"
    assert pq.codebooks.grad.abs().sum() > 0, "STE: codebook grad is zero"
    print("  [PASS] variable K + STE")

def test_variable_K_prior_floor():
    """Variable K + prior_floor: no NaN/inf."""
    K_list = [4, 8, 16, 32]
    pq = SoftPQ(G, K_list, d, lmbda=1.0, prior_floor=0.01)
    Z = torch.randn(N, G * d)
    Z_hat, usage = pq._quantise(Z)
    assert not torch.isnan(Z_hat).any(), "NaN in Z_hat"
    assert not torch.isinf(Z_hat).any(), "inf in Z_hat"
    loss = Z_hat.pow(2).sum()
    loss.backward()
    assert not torch.isnan(pq.codebooks.grad).any(), "NaN in codebook grad"
    print("  [PASS] variable K + prior_floor")

def test_allocate_K():
    """allocate_K: budget constraint + sensitivity ordering."""
    s_g = torch.tensor([1.0, 2.0, 4.0, 8.0, 0.5, 3.0, 6.0, 1.5])
    G_test = 8
    result = allocate_K(s_g, G_test, base_K=16, K_choices=(4, 8, 16, 32, 64))
    total_bits = sum(math.log2(k) for k in result)
    expected_bits = G_test * math.log2(16)
    assert abs(total_bits - expected_bits) < 1e-6, \
        f"budget violated: {total_bits} != {expected_bits}"
    s_np = s_g.numpy()
    rank = np.argsort(s_np)
    for i in range(len(rank) - 1):
        assert result[rank[i]] <= result[rank[i + 1]], \
            f"monotonicity: K[{rank[i]}]={result[rank[i]]} > K[{rank[i+1]}]={result[rank[i+1]]}"
    print(f"  [PASS] allocate_K: {result}, total_bits={total_bits:.0f}")

def test_split_merge_resize():
    """split_merge_resize: codebook shape + valid_mask update."""
    pq = SoftPQ(G, K, d, lmbda=1.0)
    with torch.no_grad():
        pq.codebooks.data = torch.randn(G, K, d)

    new_K = [8, 16, 32, 64]
    pq.split_merge_resize(new_K)
    assert pq.K_max == 64
    assert pq.codebooks.shape == (G, 64, d)
    assert pq.valid_mask.shape == (G, 64)
    assert pq.K_per_group_t.tolist() == new_K
    for g, k in enumerate(new_K):
        assert pq.valid_mask[g, :k].all()
        if k < 64:
            assert not pq.valid_mask[g, k:].any()

    Z = torch.randn(N, G * d)
    Z_hat, usage = pq._quantise(Z)
    loss = Z_hat.pow(2).sum()
    loss.backward()
    assert pq.codebooks.grad is not None
    print("  [PASS] split_merge_resize")

def test_split_merge_with_usage():
    """split_merge_resize: merge uses usage_counts to keep best centroids."""
    pq = SoftPQ(G, K, d, lmbda=1.0)
    with torch.no_grad():
        pq.codebooks.data = torch.arange(G * K * d).float().reshape(G, K, d)
    usage = torch.zeros(G, K)
    usage[0, 0] = 100
    usage[0, 5] = 50
    new_K = [4, 16, 16, 16]
    pq.split_merge_resize(new_K, usage_counts=usage)
    assert pq.codebooks.data[0, 0].sum() == torch.arange(d).float().sum()
    print("  [PASS] split_merge with usage_counts")

def test_get_prior_pmf_variable():
    """get_prior_pmf returns list of correct lengths."""
    K_list = [4, 8, 16, 32]
    pq = SoftPQ(G, K_list, d, lmbda=1.0)
    pmfs = pq.get_prior_pmf()
    assert len(pmfs) == G
    for g, k in enumerate(K_list):
        assert pmfs[g].shape == (k,), f"group {g}: shape {pmfs[g].shape} != ({k},)"
        assert abs(pmfs[g].sum() - 1.0) < 1e-5, \
            f"group {g}: PMF sum = {pmfs[g].sum()}"
    print("  [PASS] get_prior_pmf variable K")

def test_init_from_kmeans_variable():
    """init_from_kmeans with variable K."""
    K_list = [4, 8, 16, 32]
    pq = SoftPQ(G, K_list, d, lmbda=1.0)
    Z = torch.randn(N, G * d)
    pq.init_from_kmeans(Z, device='cpu', max_iter=5)
    for g, k in enumerate(K_list):
        assert pq.codebooks.data[g, :k].abs().sum() > 0, \
            f"group {g}: valid codebook entries are zero after init"
        if k < 32:
            assert pq.codebooks.data[g, k:].abs().sum() == 0, \
                f"group {g}: padded entries nonzero after init"
    print("  [PASS] init_from_kmeans variable K")

def test_sensitivity_tracking_variable():
    """Sensitivity tracking (retain_grad) works with variable K."""
    K_list = [8, 16, 32, 16]
    pq = SoftPQ(G, K_list, d, lmbda=1.0)
    pq._track_sensitivity = True
    Z = torch.randn(N, G * d, requires_grad=True)
    Z_hat, _ = pq._quantise(Z)
    loss = Z_hat.pow(2).sum()
    loss.backward()
    assert pq._z_hat_ref is not None, "z_hat_ref is None"
    assert pq._z_hat_ref.grad is not None, "z_hat_ref.grad is None"
    grad = pq._z_hat_ref.grad
    grad_g = grad.reshape(-1, G, d)
    sg = (grad_g ** 2).sum(dim=(0, 2))
    assert sg.shape == (G,)
    assert sg.min() > 0, "some group has zero sensitivity"
    print("  [PASS] sensitivity tracking with variable K")


if __name__ == '__main__':
    print("=" * 50)
    print("Variable K_g Smoke Tests")
    print("=" * 50)
    test_uniform_K_forward_backward()
    test_variable_K_forward_backward()
    test_variable_K_ste()
    test_variable_K_prior_floor()
    test_allocate_K()
    test_split_merge_resize()
    test_split_merge_with_usage()
    test_get_prior_pmf_variable()
    test_init_from_kmeans_variable()
    test_sensitivity_tracking_variable()
    print("\n" + "=" * 50)
    print("All tests passed!")
    print("=" * 50)

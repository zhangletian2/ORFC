"""Min smoke test for cls_mode='conv2' (shared E/U for CLS).

Covers:
  1. Init identity: CLS recon == input, patch recon == legacy.
  2. Shapes: encode [B,65,D], decode [B,257,D], coded_tokens==65.
  3. CLS-only recon loss backprops into analysis/synthesis.
  4. Save/reload. R-absorption consistency WITH PQ in the loop (deployment path),
     using the project convention ``seq @ R`` (E_fold=R.T@E, U_fold=R.T@U).
"""
import numpy as np
import torch

from bilinear_residual import BilinearSpatialCodec
from soft_pq import SoftPQ

D = 1024
NP = 1
NT = 257
C = D


def make(cls_mode):
    return BilinearSpatialCodec(
        D, n_prefix=NP, scale=2, grid=None,
        down="conv2", up="conv2", C=C, cls_mode=cls_mode).eval()


def make_pq(seed=0):
    torch.manual_seed(seed)
    G, K, d = 32, 8, D // 32          # G*d = D
    return SoftPQ(G, K, d, lmbda=0.0).eval()   # lmbda=0 -> no rate, hard argmin


def rand_y(B=3, seed=0):
    torch.manual_seed(seed)
    return torch.randn(B, NT, D)


def check(name, cond):
    print(f"  [{'OK' if cond else 'FAIL'}] {name}")
    assert cond, f"FAILED: {name}"


# 1. init identity
m = make("conv2")
leg = make("learned")
Y = rand_y()
seq, aux = m.encode(Y)
Yhat = m.decode(seq, aux)
cls_err = (Yhat[:, :NP] - Y[:, :NP]).norm().item()
seql, auxl = leg.encode(Y)
Yhatl = leg.decode(seql, auxl)
patch_err = (Yhat[:, NP:] - Yhatl[:, NP:]).norm().item()
print(f"  cls recon err={cls_err:.3e}  patch-vs-legacy err={patch_err:.3e}")
check("cls recon == input (init)", cls_err < 1e-4)
check("patch recon == legacy", patch_err < 1e-4)

# 2. shapes
check("encode [B,65,D]", tuple(seq.shape) == (3, NP + 64, D))
check("decode [B,257,D]", tuple(Yhat.shape) == (3, NT, D))
check("coded_tokens==65", m.coded_tokens(NT) == NP + 64)

# 3. CLS-only loss backprops (perturb off identity-init stationary point)
m2 = make("conv2").train()
with torch.no_grad():
    m2.analysis.weight += 0.01 * torch.randn_like(m2.analysis.weight)
    m2.synthesis.weight += 0.01 * torch.randn_like(m2.synthesis.weight)
seq2, aux2 = m2.encode(Y)
Yhat2 = m2.decode(seq2, aux2)
((Yhat2[:, :NP] - Y[:, :NP]) ** 2).mean().backward()
ga, gs = m2.analysis.weight.grad, m2.synthesis.weight.grad
print(f"  |grad a|={ga.abs().sum():.4f}  |grad s|={gs.abs().sum():.4f}")
check("analysis grad", ga is not None and ga.abs().sum() > 0)
check("synthesis grad", gs is not None and gs.abs().sum() > 0)

# 4a. save/reload
m3 = make("conv2")
m4 = make("conv2")
m4.load_state_dict(m3.state_dict())
m4.eval()
hat3 = m3.decode(*m3.encode(Y))
hat4 = m4.decode(*m4.encode(Y))
check("reload preserves recon", (hat3 - hat4).norm().item() < 1e-6)

# 4b. R-absorption consistency WITH PQ in the loop.
R = torch.linalg.qr(torch.randn(D, D, dtype=torch.float64,
                                generator=torch.Generator().manual_seed(1)))[0].float()
pq = make_pq(seed=2)

# Explicit path: seq @ R -> PQ -> @ R.t() -> decode
seq_e, aux_e = m3.encode(Y)
flat = seq_e.reshape(-1, D)
Z = flat @ R
Z_hat, _ = pq._quantise(Z)
seq_hat = (Z_hat @ R.t()).reshape(3, -1, D)
Yhat_explicit = m3.decode(seq_hat, aux_e)

# Absorbed path: fold R into E and U (project convention: E_fold=R.T@E, U_fold=R.T@U)
mabs = make("conv2")
with torch.no_grad():
    mabs.analysis.weight.copy_(
        torch.einsum("ij,jdhw->idhw", R.t(), mabs.analysis.weight))
    mabs.synthesis.weight.copy_(
        torch.einsum("ij,jdhw->idhw", R.t(), mabs.synthesis.weight))
seq_a, aux_a = mabs.encode(Y)
flat_a = seq_a.reshape(-1, D)
Z_hat_a, _ = pq._quantise(flat_a)
seq_hat_a = Z_hat_a.reshape(3, -1, D)
Yhat_abs = mabs.decode(seq_hat_a, aux_a)

err_abs = (Yhat_explicit - Yhat_abs).norm().item()
rf = err_abs / (Yhat_explicit.norm().item() + 1e-9)
print(f"  R-absorbed w/ PQ recon rel-diff={rf:.3e}")
check("R-absorbed (with PQ) recon == explicit", rf < 1e-4)

print("\nALL CHECKS PASSED")

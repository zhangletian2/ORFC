"""Small contracts for the entropy-aware multi-mode forward."""

import torch

from cayley import DirectOrthogonalTransform
from codec_v1 import FeatureCodecV1
from multimode_pq import MultiModeSoftPQ
from ..v12 import qhard


def _codec(device):
    torch.manual_seed(7)
    pq = MultiModeSoftPQ(2, (2, 4), 2, lmbda=0.5).to(device)
    transform = DirectOrthogonalTransform(4).to(device)
    transform.init_from_opq(torch.eye(4, device=device))
    with torch.no_grad():
        for quantizer in pq.quantizers:
            quantizer.codebooks.normal_()
            quantizer.log_prior.normal_()
    return FeatureCodecV1(pq, transform).to(device)


def test_ecvq_forward_and_rate_match_multimode():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = _codec(device).eval()
    y = torch.randn(2, 3, 4, device=device)
    for allocation in (torch.tensor([0, 1], device=device),
                       torch.tensor([1, 0], device=device)):
        decoded, _, rate = qhard.quantise(
            codec, y, allocation, return_rate=True)
        reference, _ = codec(y, modes=allocation)
        assert torch.equal(decoded, reference)
        assert torch.allclose(rate.mean(), codec.pq._last_rate, atol=1e-6)


def test_rate_objective_reaches_priors_and_codebooks():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    codec = _codec(device).train()
    y = torch.randn(2, 3, 4, device=device)
    decoded, _, rate = qhard.quantise(
        codec, y, torch.tensor([0, 1], device=device),
        codeword_temperature=0.5, return_rate=True)
    (rate.mean() * y.shape[1] + 0.5 * decoded.square().sum() / len(y)).backward()
    assert all(q.log_prior.grad is not None and q.log_prior.grad.abs().sum() > 0
               for q in codec.pq.quantizers)
    assert all(q.codebooks.grad is not None and q.codebooks.grad.abs().sum() > 0
               for q in codec.pq.quantizers)

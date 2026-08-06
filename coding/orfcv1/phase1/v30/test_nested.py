import torch
from types import SimpleNamespace

from .nested import NestedMultiModePQ, sparse_select
from .capacity_gate import mixed_exact_allocation


def test_composed_sizes_and_prefix_gradients():
    pq = NestedMultiModePQ(2, (1, 2, 3), 3)
    for stage in pq.stages:
        torch.nn.init.normal_(stage.codebooks)
    assert [pq.composed_codebook(m).shape[1] for m in range(3)] == [2, 4, 8]
    pq.composed_codebook(2).square().sum().backward()
    assert all(stage.codebooks.grad is not None for stage in pq.stages)
    assert all(float(stage.codebooks.grad.abs().sum()) > 0 for stage in pq.stages)


def test_independent_books_are_factored_exactly():
    torch.manual_seed(7)
    books = [torch.randn(2, size, 3) for size in (2, 4, 8)]
    source = SimpleNamespace(quantizers=[
        SimpleNamespace(codebooks=book) for book in books])
    pq = NestedMultiModePQ(2, (1, 2, 3), 3)
    pq.init_from_independent(source)
    for mode, target in enumerate(books):
        distance = torch.cdist(pq.composed_codebook(mode), target)
        assert float(distance.min(-1).values.max()) < 1e-5
        assert float(distance.min(-2).values.max()) < 1e-5


def test_low_mode_does_not_touch_later_stages():
    pq = NestedMultiModePQ(1, (2, 3, 4), 2)
    for stage in pq.stages:
        torch.nn.init.normal_(stage.codebooks)
    pq.composed_codebook(0).sum().backward()
    assert pq.stages[0].codebooks.grad is not None
    assert pq.stages[1].codebooks.grad is None
    assert pq.stages[2].codebooks.grad is None


def test_sparse_path_matches_full_bank():
    torch.manual_seed(3)
    pq = NestedMultiModePQ(3, (1, 2, 3), 2)
    for stage in pq.stages:
        torch.nn.init.normal_(stage.codebooks)
    sub = torch.randn(3, 7, 2)
    allocation = torch.tensor([0, 2, 1])
    bank, labels = pq.bank(sub)
    groups = torch.arange(3)
    expected = bank[allocation, groups]
    actual, actual_labels = sparse_select(pq, sub, allocation)
    assert torch.equal(actual_labels, labels[allocation, groups])
    assert torch.equal(actual, expected)


def test_mixed_allocation_is_exact_and_covers_modes():
    row = mixed_exact_allocation(32, (1, 2, 3), 64, 1)
    assert sum((1, 2, 3)[mode] for mode in row) == 64
    assert set(row.tolist()) == {0, 1, 2}

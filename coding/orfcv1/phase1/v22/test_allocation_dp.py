from .allocation_dp import topk_allocations


def test_exact_budget_and_order():
    costs = [[3, 0, 4], [3, 0, 1], [0, 2, 5]]
    result = topk_allocations(costs, [1, 2, 3], 6, topk=4)
    assert result[0] == (1.0, (1, 2, 0))
    assert all(sum([1, 2, 3][m] for m in allocation) == 6
               for _, allocation in result)
    assert result == sorted(result, key=lambda item: (item[0], item[1]))

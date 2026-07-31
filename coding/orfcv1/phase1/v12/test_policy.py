"""CPU contracts for the exact-budget allocation policy."""

import itertools
import math
import unittest

import torch

from .allocation_policy import FixedBudgetAllocationPolicy


def enumerate_distribution(logits, bits, budget, temperature):
    groups, modes = logits.shape
    feasible, scores = [], []
    for allocation in itertools.product(range(modes), repeat=groups):
        if sum(bits[mode] for mode in allocation) != budget:
            continue
        feasible.append(allocation)
        scores.append(sum(float(logits[g, mode])
                          for g, mode in enumerate(allocation)) / temperature)
    scores = torch.tensor(scores, dtype=torch.float64)
    probabilities = torch.softmax(scores, dim=0)
    return feasible, scores, probabilities


class PolicyContracts(unittest.TestCase):
    def setUp(self):
        torch.set_default_dtype(torch.float64)
        self.groups = 4
        self.bits = (1, 2, 3)
        self.budget = 8
        self.temperature = 0.7
        self.initial = torch.tensor([
            [0.3, -0.4, 0.8],
            [-0.2, 0.5, 0.1],
            [0.7, -0.1, -0.3],
            [-0.6, 0.2, 0.4],
        ])

    def tearDown(self):
        torch.set_default_dtype(torch.float32)

    def make_policy(self):
        policy = FixedBudgetAllocationPolicy(
            self.groups, self.bits, self.budget,
            init_logits=self.initial.float()).double()
        policy.logits.data.copy_(self.initial)
        return policy

    def test_dp_matches_exhaustive_distribution(self):
        policy = self.make_policy()
        dist = policy.build(self.temperature)
        feasible, scores, probabilities = enumerate_distribution(
            policy.logits.detach(), self.bits, self.budget, self.temperature)
        expected_log_z = torch.logsumexp(scores, 0)
        torch.testing.assert_close(dist.log_partition(), expected_log_z)

        allocations = torch.tensor(feasible, dtype=torch.long)
        torch.testing.assert_close(
            dist.log_prob(allocations).exp(), probabilities)

        expected_marginals = torch.zeros(self.groups, len(self.bits))
        for probability, allocation in zip(probabilities, feasible):
            for group, mode in enumerate(allocation):
                expected_marginals[group, mode] += probability
        marginals = dist.marginals()
        torch.testing.assert_close(marginals, expected_marginals)
        torch.testing.assert_close(
            marginals.sum(1), torch.ones(self.groups, dtype=marginals.dtype))
        expected_rate = (
            marginals * torch.tensor(self.bits, dtype=marginals.dtype)).sum()
        torch.testing.assert_close(
            expected_rate, torch.tensor(float(self.budget), dtype=marginals.dtype))

    def test_score_gradient_identity(self):
        policy = self.make_policy()
        dist = policy.build(self.temperature)
        allocation = dist.map_allocation()
        gradient, = torch.autograd.grad(dist.log_prob(allocation), policy.logits)
        indicator = torch.zeros_like(policy.logits)
        indicator[torch.arange(self.groups), allocation] = 1
        expected = (indicator - dist.marginals().detach()) / self.temperature
        torch.testing.assert_close(gradient, expected)
        torch.testing.assert_close(
            gradient.sum(1), torch.zeros(self.groups, dtype=gradient.dtype),
            atol=1e-12, rtol=1e-12)

    def test_paired_estimator_matches_exact_expected_gradient(self):
        policy = self.make_policy()
        dist = policy.build(self.temperature)
        feasible, _, _ = enumerate_distribution(
            policy.logits.detach(), self.bits, self.budget, self.temperature)
        allocations = torch.tensor(feasible, dtype=torch.long)
        log_probability = dist.log_prob(allocations)
        probability = log_probability.exp()
        distortion = torch.linspace(
            0.2, 2.3, len(feasible), dtype=probability.dtype).square()
        expected = (probability * distortion).sum()
        exact, = torch.autograd.grad(expected, policy.logits, retain_graph=True)

        difference = distortion[:, None] - distortion[None, :]
        log_difference = log_probability[:, None] - log_probability[None, :]
        pair_weight = probability.detach()[:, None] * probability.detach()[None, :]
        surrogate = (0.5 * pair_weight * difference * log_difference).sum()
        paired, = torch.autograd.grad(surrogate, policy.logits)
        torch.testing.assert_close(paired, exact, atol=1e-11, rtol=1e-10)

    def test_entropy_gauge_and_extreme_logits(self):
        policy = self.make_policy()
        original = policy.build(self.temperature)
        _, _, probabilities = enumerate_distribution(
            policy.logits.detach(), self.bits, self.budget, self.temperature)
        torch.testing.assert_close(
            original.entropy(), -(probabilities * probabilities.log()).sum())
        original_marginals = original.marginals().detach()
        policy.logits.data.add_(torch.tensor([[100.0], [-80.0], [30.0], [-50.0]]))
        shifted = policy.build(self.temperature)
        torch.testing.assert_close(shifted.marginals(), original_marginals)
        policy.logits.data.copy_(torch.tensor([
            [100.0, -100.0, 0.0], [-100.0, 100.0, 0.0],
            [0.0, -100.0, 100.0], [100.0, 0.0, -100.0]]))
        extreme = policy.build(self.temperature)
        self.assertTrue(bool(torch.isfinite(extreme.log_partition())))
        self.assertTrue(bool(torch.isfinite(extreme.marginals()).all()))
        self.assertTrue(bool(torch.isfinite(extreme.entropy())))

    def test_map_and_sampling_are_exact_budget(self):
        policy = self.make_policy()
        dist = policy.build(self.temperature)
        feasible, scores, probabilities = enumerate_distribution(
            policy.logits.detach(), self.bits, self.budget, self.temperature)
        expected_map = feasible[int(torch.argmax(scores))]
        self.assertEqual(tuple(dist.map_allocation().tolist()), expected_map)

        first_generator = torch.Generator().manual_seed(123)
        second_generator = torch.Generator().manual_seed(123)
        first = dist.sample(20000, generator=first_generator)
        second = dist.sample(20000, generator=second_generator)
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(bool((policy.actual_rate(first) == self.budget).all()))

        counts = {allocation: 0 for allocation in feasible}
        for allocation in first.tolist():
            counts[tuple(allocation)] += 1
        empirical = torch.tensor(
            [counts[allocation] / len(first) for allocation in feasible])
        states = len(feasible)
        tolerance = math.sqrt(math.log(2 * states / 1e-6) / (2 * len(first)))
        self.assertLessEqual(float((empirical - probabilities).abs().max()),
                             tolerance)

    def test_unreachable_budget_and_invalid_allocation_fail(self):
        with self.assertRaises(ValueError):
            FixedBudgetAllocationPolicy(3, (1, 3), 4)
        with self.assertRaises(ValueError):
            FixedBudgetAllocationPolicy(3, (1, 2.5), 6)
        with self.assertRaises(ValueError):
            FixedBudgetAllocationPolicy(3.5, (1, 2), 5)
        policy = self.make_policy()
        dist = policy.build(self.temperature)
        with self.assertRaises(ValueError):
            dist.log_prob(torch.zeros(self.groups, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()

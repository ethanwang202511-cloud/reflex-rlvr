"""Tests for the pass@k estimator and bootstrap CIs."""

from __future__ import annotations

import math

import pytest

from reflex_rlvr.eval.pass_at_k import (
    bootstrap_pass_at_k_ci,
    mean_pass_at_k,
    mean_pass_at_k_naive_plugin,
    paired_bootstrap_delta,
    pass_at_k_estimator,
    pass_at_k_per_problem,
    yue_crossover_check,
)


class TestPassAtKEstimator:
    def test_zero_correct_zero_pass(self) -> None:
        # n=10 samples, c=0 correct, k=5: pass@5 = 0
        assert pass_at_k_estimator(n=10, c=0, k=5) == 0.0

    def test_all_correct_pass_one(self) -> None:
        assert pass_at_k_estimator(n=10, c=10, k=5) == 1.0

    def test_one_correct_pass_at_k_equals_k_over_n(self) -> None:
        # c=1, n=10, k=1: pass@1 = 1/10
        assert math.isclose(
            pass_at_k_estimator(n=10, c=1, k=1), 0.1, abs_tol=1e-9
        )

    def test_k_greater_than_n_returns_zero(self) -> None:
        assert pass_at_k_estimator(n=4, c=2, k=10) == 0.0

    def test_invalid_k(self) -> None:
        with pytest.raises(ValueError):
            pass_at_k_estimator(n=10, c=5, k=0)

    def test_invalid_c(self) -> None:
        with pytest.raises(ValueError):
            pass_at_k_estimator(n=10, c=11, k=5)


class TestPerProblem:
    def test_per_problem_simple(self) -> None:
        records = [[1, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1]]
        out = pass_at_k_per_problem(records, k=2)
        # Problem 0: c=1/4, k=2 → 1 - C(3,2)/C(4,2) = 1 - 3/6 = 0.5
        # Problem 1: c=0/4, k=2 → 0
        # Problem 2: all correct → 1
        assert math.isclose(out[0], 0.5, abs_tol=1e-9)
        assert math.isclose(out[1], 0.0, abs_tol=1e-9)
        assert math.isclose(out[2], 1.0, abs_tol=1e-9)

    def test_mean_pass_at_k(self) -> None:
        records = [[1, 0], [0, 1], [1, 1], [0, 0]]
        m = mean_pass_at_k(records, k=1)
        # Each pass@1 = c/n: 0.5, 0.5, 1.0, 0.0; mean = 0.5
        assert math.isclose(m, 0.5, abs_tol=1e-9)


class TestBootstrap:
    def test_ci_brackets_point(self) -> None:
        records = [[1] * 10 for _ in range(20)]  # all-correct
        mean, lo, hi = bootstrap_pass_at_k_ci(records, k=1, seed=0)
        assert mean == 1.0
        assert lo <= mean <= hi

    def test_paired_bootstrap_zero_diff(self) -> None:
        records = [[1, 0, 1, 0] for _ in range(10)]
        delta, lo, hi, p = paired_bootstrap_delta(
            records, records, k=1, seed=0
        )
        assert delta == 0.0
        assert lo <= 0 <= hi

    def test_paired_bootstrap_a_better(self) -> None:
        a = [[1, 1, 1, 1] for _ in range(20)]
        b = [[0, 0, 0, 0] for _ in range(20)]
        delta, lo, hi, _ = paired_bootstrap_delta(a, b, k=1, seed=0)
        assert delta == 1.0
        assert lo > 0


class TestYueCrossoverCheck:
    def test_method_strictly_better(self) -> None:
        base = [[0, 0, 0, 0] for _ in range(10)]
        meth = [[1, 0, 0, 0] for _ in range(10)]
        out = yue_crossover_check(base, meth, ks=(1, 2, 4))
        for k in (1, 2, 4):
            assert out[k]["delta"] > 0

    def test_method_no_advantage(self) -> None:
        base = [[1, 0, 0, 0] for _ in range(10)]
        meth = [[1, 0, 0, 0] for _ in range(10)]
        out = yue_crossover_check(base, meth)
        for v in out.values():
            assert v["delta"] == 0.0


class TestNaivePlugin:
    def test_matches_chen_when_p_hat_extreme(self) -> None:
        # When p̂ ∈ {0, 1} the plug-in agrees with Chen et al.
        records = [[0, 0, 0, 0], [1, 1, 1, 1]]
        chen = mean_pass_at_k(records, k=4)
        naive = mean_pass_at_k_naive_plugin(records, k=4)
        assert math.isclose(chen, naive, abs_tol=1e-9)

    def test_naive_biased_relative_to_chen(self) -> None:
        # Pick a problem where the two estimators differ.
        records = [[1, 0, 0, 0]]  # c=1, n=4
        # Chen pass@2 = 1 - C(3,2)/C(4,2) = 1 - 3/6 = 0.5
        # Naive: p̂=0.25, pass@2 = 1 - 0.75^2 = 0.4375
        chen = mean_pass_at_k(records, k=2)
        naive = mean_pass_at_k_naive_plugin(records, k=2)
        assert chen != naive
        assert math.isclose(chen, 0.5, abs_tol=1e-9)

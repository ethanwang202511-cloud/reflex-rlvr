"""Tests for the cosine-anneal noise schedule."""

from __future__ import annotations

import math

import pytest

from reflex_rlvr.latent.cosine_anneal import (
    cosine_anneal_noise,
    schedule_curve,
)


class TestCosineAnnealNoise:
    def test_step_zero_returns_eps_max(self) -> None:
        eps = cosine_anneal_noise(0, S_max=32, eps_max=0.10)
        assert math.isclose(eps, 0.10, abs_tol=1e-9)

    def test_step_S_max_returns_zero(self) -> None:
        eps = cosine_anneal_noise(32, S_max=32, eps_max=0.10)
        assert math.isclose(eps, 0.0, abs_tol=1e-9)

    def test_step_S_max_over_2_is_eps_max_over_2(self) -> None:
        eps = cosine_anneal_noise(16, S_max=32, eps_max=0.10)
        assert math.isclose(eps, 0.05, abs_tol=1e-9)

    def test_monotonic_decrease(self) -> None:
        curve = schedule_curve(S_max=32, eps_max=0.10)
        for prev, curr in zip(curve, curve[1:]):
            assert prev >= curr - 1e-12

    def test_clamps_outside_range(self) -> None:
        # Below 0 clamps to eps_max
        assert cosine_anneal_noise(-5, S_max=32, eps_max=0.10) == pytest.approx(
            0.10
        )
        # Above S_max clamps to 0
        assert cosine_anneal_noise(40, S_max=32, eps_max=0.10) == pytest.approx(
            0.0, abs=1e-9
        )

    def test_invalid_S_max(self) -> None:
        with pytest.raises(ValueError):
            cosine_anneal_noise(0, S_max=0, eps_max=0.1)
        with pytest.raises(ValueError):
            cosine_anneal_noise(0, S_max=-1, eps_max=0.1)

    def test_invalid_eps_max(self) -> None:
        with pytest.raises(ValueError):
            cosine_anneal_noise(0, S_max=32, eps_max=-0.01)

    def test_per_cycle_eps_max_staircase(self) -> None:
        # Architecture §3.1: cycle 1 = 0.10, cycle 2 = 0.15, cycles 3-5 = 0.20.
        # Confirm those produce different curves at the same step.
        cyc1 = cosine_anneal_noise(0, S_max=32, eps_max=0.10)
        cyc2 = cosine_anneal_noise(0, S_max=32, eps_max=0.15)
        cyc3 = cosine_anneal_noise(0, S_max=32, eps_max=0.20)
        assert cyc1 < cyc2 < cyc3

"""Tests for latent / halting / post-block diagnostics."""

from __future__ import annotations

import math

import pytest
import torch

from reflex_rlvr.latent.diagnostics import (
    halting_entropy,
    halting_short_circuit_alarm,
    latent_diversity_alarm,
    latent_first_step_entropy,
    post_block_ppl_ratio,
)


class TestLatentFirstStepEntropy:
    def test_uniform_logits_max_entropy(self) -> None:
        n_problems, n_rollouts, vocab = 2, 3, 8
        logits = torch.zeros(n_problems, n_rollouts, vocab)
        ent = latent_first_step_entropy(logits)
        assert math.isclose(ent, math.log(vocab), abs_tol=1e-5)

    def test_one_hot_logits_zero_entropy(self) -> None:
        logits = torch.full((1, 1, 4), -1e9)
        logits[0, 0, 2] = 1e9
        ent = latent_first_step_entropy(logits)
        assert ent < 1e-3

    def test_wrong_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            latent_first_step_entropy(torch.zeros(4, 8))


class TestHaltingEntropy:
    def test_uniform_distribution(self) -> None:
        steps = list(range(1, 9))  # 8 distinct halt steps, each once
        ent = halting_entropy(steps)
        assert math.isclose(ent, math.log(8), abs_tol=1e-9)

    def test_concentrated_distribution(self) -> None:
        steps = [3] * 100  # always halts at step 3
        ent = halting_entropy(steps)
        assert ent == 0.0

    def test_empty_returns_zero(self) -> None:
        assert halting_entropy([]) == 0.0

    def test_tensor_input_works(self) -> None:
        ent = halting_entropy(torch.tensor([2, 2, 3, 3]))
        assert math.isclose(ent, math.log(2), abs_tol=1e-9)


class TestPostBlockPPLRatio:
    def test_ratio_one_when_logits_perfect(self) -> None:
        # Logits assign all mass to the actual token → PPL = 1.0
        vocab = 5
        T = 4
        logits = torch.full((T, vocab), -1e9)
        token_ids = torch.tensor([0, 1, 2, 3])
        for t, tid in enumerate(token_ids):
            logits[t, int(tid)] = 1e9
        # Base PPL = 1.0 → ratio = 1.0
        r = post_block_ppl_ratio(logits, token_ids, base_ppl=1.0)
        assert math.isclose(r, 1.0, abs_tol=1e-3)

    def test_ratio_high_when_logits_random(self) -> None:
        torch.manual_seed(0)
        vocab = 100
        T = 8
        logits = torch.randn(T, vocab) * 0.1
        token_ids = torch.randint(0, vocab, (T,))
        # Base PPL = 1.0 (artificial) → ratio should be much greater than 1.
        r = post_block_ppl_ratio(logits, token_ids, base_ppl=1.0)
        assert r > 5.0

    def test_wrong_shape_raises(self) -> None:
        with pytest.raises(ValueError):
            post_block_ppl_ratio(torch.zeros(5), torch.tensor([0, 1]), base_ppl=1.0)


class TestLatentDiversityAlarm:
    def test_no_alarm_when_stable(self) -> None:
        d = {1: 5.0, 2: 4.9, 3: 4.85, 4: 4.8}
        fired, _ = latent_diversity_alarm(d)
        assert not fired

    def test_alarm_fires_on_30pct_drop(self) -> None:
        d = {1: 5.0, 2: 5.0, 3: 3.0}
        fired, reason = latent_diversity_alarm(d)
        assert fired
        assert "cycle 3" in reason

    def test_handles_zero_baseline(self) -> None:
        # Should not divide by zero
        d = {1: 0.0, 2: 0.5}
        fired, _ = latent_diversity_alarm(d)
        assert not fired


class TestHaltingShortCircuitAlarm:
    def test_no_alarm_when_above_threshold(self) -> None:
        d = {1: 1.5, 2: 1.4, 3: 1.3}
        fired, _ = halting_short_circuit_alarm(d)
        assert not fired

    def test_alarm_fires_at_cycle_2(self) -> None:
        d = {1: 1.5, 2: 0.2}
        fired, reason = halting_short_circuit_alarm(d)
        assert fired
        assert "cycle 2" in reason

    def test_does_not_fire_at_cycle_1(self) -> None:
        # Cycle 1 is exempt from the early-cycle threshold per
        # proposal §2.7.4.
        d = {1: 0.1}
        fired, _ = halting_short_circuit_alarm(d, earliest_cycle=2)
        assert not fired

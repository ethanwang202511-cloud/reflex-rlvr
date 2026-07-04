"""Tests for the halting head + heuristic warm-up target + value head."""

from __future__ import annotations

import torch

from reflex_rlvr.latent.halt_head import (
    HaltHead,
    ValueHead,
    heuristic_warmup_target,
    sinusoidal_step_encoding,
)


class TestSinusoidalEncoding:
    def test_dim_matches(self) -> None:
        enc = sinusoidal_step_encoding(3, dim=16)
        assert enc.shape == (16,)

    def test_odd_dim_padded_with_zero(self) -> None:
        enc = sinusoidal_step_encoding(5, dim=15)
        assert enc.shape == (15,)
        # Last position should be the zero pad
        assert float(enc[-1].item()) == 0.0

    def test_batched_input(self) -> None:
        enc = sinusoidal_step_encoding(torch.tensor([1, 2, 3]), dim=8)
        assert enc.shape == (3, 8)

    def test_different_steps_give_different_encodings(self) -> None:
        e0 = sinusoidal_step_encoding(0, dim=16)
        e1 = sinusoidal_step_encoding(1, dim=16)
        assert not torch.allclose(e0, e1)


class TestHaltHead:
    def test_forward_shape(self) -> None:
        head = HaltHead(d_model=64, step_dim=8)
        h = torch.randn(4, 64)
        out = head(h, s=2)
        assert out.shape == (4,)

    def test_forward_unbatched_input(self) -> None:
        head = HaltHead(d_model=64, step_dim=8)
        h = torch.randn(64)
        out = head(h, s=2)
        assert out.shape == (1,)

    def test_returns_finite_logits(self) -> None:
        torch.manual_seed(0)
        head = HaltHead(d_model=64, step_dim=8)
        h = torch.randn(8, 64)
        out = head(h, s=4)
        assert torch.isfinite(out).all()

    def test_default_h_norm_uses_actual_norm(self) -> None:
        torch.manual_seed(0)
        head = HaltHead(d_model=32, step_dim=8)
        h = torch.randn(2, 32)
        # Provide explicit norm equal to actual; should match default
        explicit = head(h, s=3, h_s_norm=h.norm(dim=-1))
        default = head(h, s=3)
        assert torch.allclose(explicit, default, atol=1e-6)

    def test_works_with_eval_mode(self) -> None:
        head = HaltHead(d_model=32, step_dim=8)
        head.eval()
        with torch.no_grad():
            h = torch.randn(3, 32)
            out = head(h, s=5)
            assert out.shape == (3,)


class TestHeuristicWarmupTarget:
    def test_no_halt_when_above_threshold(self) -> None:
        # Δh / h = 0.5 always; never halts.
        target = heuristic_warmup_target(
            delta_h_norms=[0.5, 0.5, 0.5, 0.5],
            h_norms=[1.0, 1.0, 1.0, 1.0],
            relative_threshold=0.05,
        )
        assert target == [0, 0, 0, 0]

    def test_halts_after_two_consec_below(self) -> None:
        target = heuristic_warmup_target(
            delta_h_norms=[0.5, 0.5, 0.01, 0.01, 0.5],
            h_norms=[1.0, 1.0, 1.0, 1.0, 1.0],
            relative_threshold=0.05,
        )
        # Step 3 has 2 consecutive low ratios → halt label.
        assert target == [0, 0, 0, 1, 0]

    def test_consec_resets_on_violation(self) -> None:
        target = heuristic_warmup_target(
            delta_h_norms=[0.01, 0.5, 0.01, 0.01],
            h_norms=[1.0, 1.0, 1.0, 1.0],
            relative_threshold=0.05,
        )
        assert target == [0, 0, 0, 1]

    def test_empty_input(self) -> None:
        assert heuristic_warmup_target([], []) == []


class TestValueHead:
    def test_forward_shape(self) -> None:
        v = ValueHead(d_model=32)
        out = v(torch.randn(5, 32))
        assert out.shape == (5,)

    def test_forward_unbatched(self) -> None:
        v = ValueHead(d_model=32)
        out = v(torch.randn(32))
        assert out.shape == (1,)

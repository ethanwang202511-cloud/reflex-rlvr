"""Tests for the Gradient-Spectral Initialization primitive."""

from __future__ import annotations

import torch

from reflex_rlvr.gsi.gradient_spectral_init import (
    gradient_spectral_init,
    norm_match,
    regularize_off_subspace,
    top_k_eigvecs_of_gradient_covariance,
)


class TestNormMatch:
    def test_target_norm(self) -> None:
        v = torch.tensor([3.0, 4.0])  # norm 5
        out = norm_match(v, target_norm=10.0)
        assert torch.isclose(out.norm(), torch.tensor(10.0))

    def test_zero_vec_unchanged(self) -> None:
        v = torch.zeros(4)
        out = norm_match(v, target_norm=10.0)
        assert torch.equal(v, out)


class TestTopKEigvecs:
    def test_shape(self) -> None:
        torch.manual_seed(0)
        G = torch.randn(100, 16)
        vecs, vals = top_k_eigvecs_of_gradient_covariance(G, k=4)
        assert vecs.shape == (16, 4)
        assert vals.shape == (4,)

    def test_descending_eigenvalues(self) -> None:
        torch.manual_seed(0)
        G = torch.randn(50, 8)
        _, vals = top_k_eigvecs_of_gradient_covariance(G, k=4)
        for prev, curr in zip(vals.tolist(), vals.tolist()[1:]):
            assert prev >= curr - 1e-6

    def test_orthonormal_eigvecs(self) -> None:
        torch.manual_seed(0)
        G = torch.randn(100, 8)
        vecs, _ = top_k_eigvecs_of_gradient_covariance(G, k=4)
        # Columns should be orthonormal
        gram = vecs.T @ vecs
        assert torch.allclose(gram, torch.eye(4), atol=1e-5)


class TestRegularizeOffSubspace:
    def test_off_subspace_noise_orthogonal(self) -> None:
        torch.manual_seed(0)
        d, k = 16, 4
        # Build a random orthonormal subspace basis
        Q = torch.linalg.qr(torch.randn(d, k))[0]
        v = Q @ torch.randn(k)  # in subspace
        out = regularize_off_subspace(
            v, Q, pct=0.05, target_norm=v.norm().item()
        )
        # Project (out - v) onto subspace; should be near zero
        diff = out - v
        proj_back = Q @ (Q.T @ diff)
        assert torch.allclose(proj_back, torch.zeros_like(diff), atol=1e-4)


class TestGradientSpectralInit:
    def test_shape_and_finite(self) -> None:
        torch.manual_seed(0)
        d = 32
        G = torch.randn(200, d)
        target_norm = 1.5
        embeds = gradient_spectral_init(
            G,
            n_special_tokens=3,
            k=8,
            target_norm=target_norm,
        )
        assert embeds.shape == (3, d)
        assert torch.isfinite(embeds).all()

    def test_norm_close_to_target(self) -> None:
        torch.manual_seed(0)
        d = 32
        G = torch.randn(200, d)
        target_norm = 1.5
        embeds = gradient_spectral_init(
            G,
            n_special_tokens=3,
            k=8,
            target_norm=target_norm,
            off_subspace_pct=0.05,
        )
        # Each embedding has on-subspace component norm-matched to
        # target, plus 5% off-subspace noise; total norm is between
        # target and target * sqrt(1 + pct^2). Allow generous slack.
        for i in range(3):
            n = float(embeds[i].norm().item())
            assert 0.8 * target_norm <= n <= 1.2 * target_norm

    def test_reproducible_with_generator(self) -> None:
        d = 32
        G = torch.randn(200, d)
        gen1 = torch.Generator().manual_seed(42)
        gen2 = torch.Generator().manual_seed(42)
        e1 = gradient_spectral_init(
            G, n_special_tokens=3, k=8, target_norm=1.0, generator=gen1
        )
        e2 = gradient_spectral_init(
            G, n_special_tokens=3, k=8, target_norm=1.0, generator=gen2
        )
        assert torch.allclose(e1, e2)

    def test_different_seeds_different_outputs(self) -> None:
        d = 32
        G = torch.randn(200, d)
        gen1 = torch.Generator().manual_seed(0)
        gen2 = torch.Generator().manual_seed(1)
        e1 = gradient_spectral_init(
            G, n_special_tokens=3, k=8, target_norm=1.0, generator=gen1
        )
        e2 = gradient_spectral_init(
            G, n_special_tokens=3, k=8, target_norm=1.0, generator=gen2
        )
        assert not torch.allclose(e1, e2)

    def test_invalid_k(self) -> None:
        import pytest

        G = torch.randn(10, 8)
        with pytest.raises(ValueError):
            gradient_spectral_init(
                G, n_special_tokens=3, k=0, target_norm=1.0
            )

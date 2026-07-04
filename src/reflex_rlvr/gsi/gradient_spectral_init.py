"""Gradient-Spectral Initialization for new-token embeddings.

Architecture §1.2.1 (iter-D3-corrected algorithm).

The audit-round-2 fix: gradients are taken with respect to
*intermediate* CoT-token logits, not the final answer-token logit.
The intuition: directions that consistently produce gradient signal
during *next-step reasoning prediction* are causally relevant for the
iterative computation the latent register performs. Gradients of the
final answer-token logit are answer-extraction directions
(approximately the unembedding rows for the answer token) — seeding
the latent register with those would bias it toward emitting the
answer immediately, the *opposite* of the intended effect.

The numerical primitive (top-k eigenvectors of a gradient covariance,
norm-matched + 5% off-subspace regularizer) is implemented here in a
model-agnostic way; the calibration loop that actually computes the
gradients on a real LLM lives in the calling script (``scripts/``)
because it depends on a particular model's residual-stream API.

This file is import-safe with no GPU dependencies.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


def norm_match(vec: torch.Tensor, target_norm: float) -> torch.Tensor:
    """Rescale ``vec`` so its L2 norm equals ``target_norm``."""
    n = vec.norm()
    if n < 1e-12:
        return vec
    return vec * (target_norm / n)


def regularize_off_subspace(
    vec: torch.Tensor,
    subspace_basis: torch.Tensor,
    *,
    pct: float = 0.05,
    target_norm: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Add ``pct`` * target_norm of Gaussian noise *orthogonal* to the
    spectral subspace, so AdamW updates are well-conditioned in the
    orthogonal directions.

    Parameters
    ----------
    vec:
        The on-subspace component (1-D tensor of length d).
    subspace_basis:
        Tensor of shape (d, k); the spectral subspace.
    pct:
        Fraction of the target norm to allocate to off-subspace noise.
        Architecture §1.2.1 uses 0.05.
    target_norm:
        Norm to use for the noise scaling. Defaults to ``vec.norm()``.
    generator:
        Optional torch.Generator for reproducible noise.
    """
    if vec.dim() != 1:
        raise ValueError(f"expected 1-D vec, got shape {tuple(vec.shape)}")
    if subspace_basis.dim() != 2 or subspace_basis.shape[0] != vec.shape[0]:
        raise ValueError(
            f"expected subspace_basis (d, k); got shape "
            f"{tuple(subspace_basis.shape)}"
        )

    target_norm = target_norm if target_norm is not None else float(vec.norm().item())
    noise = torch.randn(
        vec.shape, generator=generator, dtype=vec.dtype, device=vec.device
    )
    # Project out the subspace component
    proj = subspace_basis @ (subspace_basis.T @ noise)
    noise_off = noise - proj
    n_off = noise_off.norm()
    if n_off < 1e-12:
        return vec
    return vec + pct * target_norm * noise_off / n_off


def top_k_eigvecs_of_gradient_covariance(
    grad_matrix: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute the top-k eigenvectors (and eigenvalues) of
    ``G^T G / N`` where rows of ``grad_matrix`` are gradient samples.

    Parameters
    ----------
    grad_matrix:
        Tensor of shape (N, d). Each row is one gradient sample.
    k:
        Number of top eigenvectors to keep.

    Returns
    -------
    (top_k_eigvecs, top_k_eigvals):
        ``top_k_eigvecs`` shape (d, k), columns are eigenvectors,
        sorted by descending eigenvalue.
        ``top_k_eigvals`` shape (k,), descending.
    """
    if grad_matrix.dim() != 2:
        raise ValueError(
            f"expected (N, d) grad_matrix; got shape {tuple(grad_matrix.shape)}"
        )
    n, d = grad_matrix.shape
    if n == 0:
        raise ValueError("grad_matrix must have at least one row")
    if k > d:
        raise ValueError(f"k={k} cannot exceed d={d}")

    cov = (grad_matrix.float().T @ grad_matrix.float()) / n  # (d, d)
    eigvals, eigvecs = torch.linalg.eigh(cov)  # ascending
    top_k_vecs = eigvecs[:, -k:].flip(-1)  # descending
    top_k_vals = eigvals[-k:].flip(-1)
    return top_k_vecs, top_k_vals


def gradient_spectral_init(
    grad_matrix: torch.Tensor,
    *,
    n_special_tokens: int,
    k: int = 64,
    target_norm: float,
    off_subspace_pct: float = 0.05,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Compose the GSI v2 algorithm: top-k eigvecs → randomized linear
    combinations → norm-match → off-subspace regularizer.

    Parameters
    ----------
    grad_matrix:
        (N, d) — gradient samples at the tap layer w.r.t. intermediate
        CoT-token logits. Rows can be from any number of calibration
        problems / token positions; only the covariance matters.
    n_special_tokens:
        Number of new-token embeddings to seed (3 in REFLEX-RLVR:
        ``<think>``, ``</think>``, ``<latent>``).
    k:
        Spectral subspace dimensionality. Architecture §1.2.1 uses 64.
    target_norm:
        Mean L2 norm of the existing token embeddings — the new
        embeddings are scaled to match.
    off_subspace_pct:
        Fraction of ``target_norm`` allocated to off-subspace noise.
        Architecture §1.2.1 uses 0.05.
    generator:
        Optional ``torch.Generator`` for reproducibility.

    Returns
    -------
    Tensor of shape (n_special_tokens, d). Each row is an embedding for
    one new special token.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if n_special_tokens <= 0:
        raise ValueError(
            f"n_special_tokens must be positive, got {n_special_tokens}"
        )

    top_k_vecs, top_k_vals = top_k_eigvecs_of_gradient_covariance(
        grad_matrix, k=k
    )
    d = grad_matrix.shape[1]
    new_embeddings = torch.zeros(n_special_tokens, d, dtype=torch.float32)

    # Use eigvalues' sqrt as the per-direction prior on weight
    # magnitude, mirroring the audit-corrected pseudo-code in
    # architecture §1.2.1.
    eigvals_sqrt = top_k_vals.clamp_min(0.0).sqrt()

    for i in range(n_special_tokens):
        weights = (
            torch.randn(k, generator=generator, dtype=torch.float32)
            * eigvals_sqrt
        )
        on_subspace = top_k_vecs.float() @ weights  # (d,)
        on_subspace = norm_match(on_subspace, target_norm)
        new_embeddings[i] = regularize_off_subspace(
            on_subspace,
            top_k_vecs.float(),
            pct=off_subspace_pct,
            target_norm=target_norm,
            generator=generator,
        )

    return new_embeddings


def collect_intermediate_grads(
    model: torch.nn.Module,
    calibration_problems: list[dict],
    *,
    tap_layer: int,
    n_positions_per_problem: int = 32,
    forward_with_residuals: Callable[..., dict] | None = None,
) -> torch.Tensor:
    """Skeleton of the calibration loop. The actual implementation is
    LLM-specific (it depends on a model exposing per-layer residual
    streams), so the calling script is expected to provide
    ``forward_with_residuals``. This function provides the gradient-
    matrix-assembly logic; it does not implement the model forward.

    Each calibration problem dict has keys ``x_ids`` (prompt tokens),
    ``cot_ids`` (CoT tokens), and ``answer_id`` (final-answer token);
    we sample 32 positions inside the CoT (excluding the final answer
    position) and compute gradients of the next-token logit w.r.t.
    the residual stream at the tap layer.

    Returns ``grad_matrix`` of shape (N_problems × n_positions, d).
    """
    if forward_with_residuals is None:
        raise NotImplementedError(
            "forward_with_residuals callback must be provided; calibration "
            "is model-specific and lives in the calling script."
        )

    grads = []
    for prob in calibration_problems:
        x_ids = prob["x_ids"]
        cot_ids = prob["cot_ids"]
        cot_start = len(x_ids)
        cot_end = cot_start + max(0, len(cot_ids) - 1)  # exclude final answer
        if cot_end <= cot_start:
            continue
        positions = torch.randint(
            cot_start, cot_end, (n_positions_per_problem,)
        ).tolist()

        # Caller-provided forward — see scripts/run_gsi_calibration.py
        problem_grads = forward_with_residuals(
            x_ids=x_ids,
            cot_ids=cot_ids,
            tap_layer=tap_layer,
            positions=positions,
        )
        grads.append(problem_grads.detach())

    if not grads:
        raise RuntimeError("no gradients collected; calibration set empty")
    return torch.cat(grads, dim=0)

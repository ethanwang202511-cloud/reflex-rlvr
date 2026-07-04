"""Gradient-Spectral Initialization (GSI) — architecture §1.2.1.

Solves the new-token cold-start problem for ``<think>``, ``</think>``,
``<latent>`` embeddings. Random init produces a 0.3% ``<think>``-
emission rate after 500 RL steps; GSI is predicted to lift this to
≥ 30%. If the lift is < 10×, GSI is null (architecture §1.2.1) and we
fall back to random-init + heuristic warm-up.
"""

from reflex_rlvr.gsi.gradient_spectral_init import (
    gradient_spectral_init,
    regularize_off_subspace,
    norm_match,
)

__all__ = [
    "gradient_spectral_init",
    "regularize_off_subspace",
    "norm_match",
]

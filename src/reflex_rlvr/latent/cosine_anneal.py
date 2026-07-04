"""Cosine-anneal noise schedule for the latent register.

Per architecture §2.1 / §2.3 — the noise schedule replaces the pre-cut
criticality head (proposal §7.5.0).

    eps_s = eps_max * 0.5 * (1 + cos(pi * s / S_max))

at step s in [0, S_max]. High noise on early latent steps (the
"uncommitted" steps where the residual stream has the most room to
move); zero noise at s = S_max (preserve fidelity of the post-block
discrete context).
"""

from __future__ import annotations

import math


def cosine_anneal_noise(
    s: int | float,
    *,
    S_max: int = 32,
    eps_max: float = 0.10,
) -> float:
    """Return the noise variance multiplier eps_s at step s.

    Parameters
    ----------
    s:
        Latent step index, 0 ≤ s ≤ S_max. Outside this interval the
        function clamps and returns the boundary value.
    S_max:
        Maximum latent steps per ``<think>`` block. Defaults to 32 per
        architecture §3.1.
    eps_max:
        Per-cycle peak noise variance. Defaults to 0.10 (cycle-1
        value); cycle 2 uses 0.15, cycles 3–5 use 0.20.
    """
    if S_max <= 0:
        raise ValueError(f"S_max must be positive, got {S_max}")
    if eps_max < 0:
        raise ValueError(f"eps_max must be non-negative, got {eps_max}")

    s_clamped = max(0.0, min(float(s), float(S_max)))
    return eps_max * 0.5 * (1.0 + math.cos(math.pi * s_clamped / S_max))


def schedule_curve(S_max: int = 32, eps_max: float = 0.10) -> list[float]:
    """Helper for plotting / sanity checks. Returns the full curve."""
    return [cosine_anneal_noise(s, S_max=S_max, eps_max=eps_max) for s in range(S_max + 1)]

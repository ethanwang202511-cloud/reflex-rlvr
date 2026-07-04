"""Halting head — decides when to emit ``</think>``.

Architecture §2.2 / §4.2:

- Input: residual stream ``h_s`` (shape (B, d_model)), step index s
  (sinusoidal encoded), and ``‖Δh_s‖`` (the change in residual stream
  from the previous latent step).
- Output: a single logit; halt if ``sigmoid(logit) > 0.5`` AND
  ``s ≥ S_min``.
- Two-stage training: cross-entropy warm-up against a heuristic target
  for the first 500 RL steps, then PPO with a value head for the rest
  of the run. The value head is jointly updated and discarded at
  inference.

This module owns the *architecture* (small enough to unit-test on Mac
CPU). The training loop lives in ``modal_app/`` and is wired against
the trainer's PPO implementation.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def sinusoidal_step_encoding(s: int | torch.Tensor, dim: int) -> torch.Tensor:
    """Standard transformer-style sinusoidal encoding of a scalar step.

    Returns a tensor of shape (dim,). Vectorized over a 1-D tensor of
    steps (returns shape (B, dim)).
    """
    if isinstance(s, int):
        s_t = torch.tensor([s], dtype=torch.float32)
    else:
        s_t = s.to(dtype=torch.float32).reshape(-1)

    half = dim // 2
    if half == 0:
        return s_t.unsqueeze(-1)

    freqs = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, dtype=torch.float32)
        / max(1, half - 1)
    )
    args = s_t.unsqueeze(-1) * freqs.unsqueeze(0)  # (B, half)
    enc = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        enc = torch.cat([enc, torch.zeros_like(enc[..., :1])], dim=-1)
    if isinstance(s, int):
        return enc.squeeze(0)
    return enc


class HaltHead(nn.Module):
    """Two-layer halting MLP with SiLU activation.

    Inputs concatenated:
        h_s             : (B, d_model)
        s_emb           : (B, step_dim)              sinusoidal
        h_s_norm        : (B, 1)                     ‖h_s‖_2

    Output: (B,) — single halting logit per example.

    The halting *probability* is sigmoid of the logit; we follow
    architecture §2.1 in halting iff sigmoid(logit) > 0.5 AND s ≥ S_min.
    """

    def __init__(
        self,
        d_model: int,
        *,
        step_dim: int = 16,
        hidden_div: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.step_dim = step_dim
        in_dim = d_model + step_dim + 1
        hidden = max(1, d_model // hidden_div)
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        h_s: torch.Tensor,
        s: int | torch.Tensor,
        h_s_norm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if h_s.dim() == 1:
            h_s = h_s.unsqueeze(0)  # (1, d_model)
        batch_size = h_s.shape[0]

        if h_s_norm is None:
            h_s_norm = h_s.norm(dim=-1, keepdim=True)  # (B, 1)
        elif h_s_norm.dim() == 0:
            h_s_norm = h_s_norm.expand(batch_size, 1)
        elif h_s_norm.dim() == 1:
            h_s_norm = h_s_norm.unsqueeze(-1)

        s_emb = sinusoidal_step_encoding(s, self.step_dim).to(h_s.device)
        if s_emb.dim() == 1:
            s_emb = s_emb.unsqueeze(0).expand(batch_size, -1)

        x = torch.cat([h_s.float(), s_emb.float(), h_s_norm.float()], dim=-1)
        return self.proj(x).squeeze(-1)


def heuristic_warmup_target(
    delta_h_norms: list[float],
    h_norms: list[float],
    *,
    relative_threshold: float = 0.05,
    consec_required: int = 2,
) -> list[int]:
    """Heuristic warm-up label — halt when ``‖Δh_s‖ / ‖h_s‖`` < threshold
    for ``consec_required`` consecutive steps. Used as the cross-entropy
    target during steps 0–500.

    Returns a per-step binary halt label (1 if the heuristic fires *at*
    this step). Consistent with architecture §2.2 / §4.2.
    """
    if len(delta_h_norms) != len(h_norms):
        raise ValueError("delta_h_norms and h_norms must have equal length")

    labels = [0] * len(delta_h_norms)
    consec = 0
    for i, (dh, h) in enumerate(zip(delta_h_norms, h_norms)):
        ratio = dh / max(h, 1e-9)
        if ratio < relative_threshold:
            consec += 1
        else:
            consec = 0
        if consec >= consec_required:
            labels[i] = 1
    return labels


class ValueHead(nn.Module):
    """Value head used jointly with the halting head during PPO.

    Architecture §2.2: 2-layer MLP, SiLU, predicts the per-step return
    ``E[r_τ - λ_step · n_remaining_steps | h_s]`` via Huber loss.
    Discarded at inference.
    """

    def __init__(self, d_model: int, *, hidden_div: int = 4) -> None:
        super().__init__()
        hidden = max(1, d_model // hidden_div)
        self.proj = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h_s: torch.Tensor) -> torch.Tensor:
        if h_s.dim() == 1:
            h_s = h_s.unsqueeze(0)
        return self.proj(h_s.float()).squeeze(-1)

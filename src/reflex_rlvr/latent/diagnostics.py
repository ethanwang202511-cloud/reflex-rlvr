"""Latent-register diagnostics: entropies, post-block PPL.

Architecture §5.2.1 (latent first-step entropy, halting entropy) and
§2.3.1 (post-block PPL diagnostic).

These are the small numerical primitives. Hooks into the model
forward pass live in ``modal_app/`` — here we own the math so it can
be unit-tested on Mac CPU with synthetic distributions.
"""

from __future__ import annotations

from collections import Counter

import torch


def _stable_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(logits.float(), dim=dim)


def latent_first_step_entropy(
    first_step_logits: torch.Tensor,
) -> float:
    """Average per-rollout entropy of the policy's first latent-step
    token-distribution (proposal §2.7.3).

    Parameters
    ----------
    first_step_logits:
        Tensor of shape (n_problems, n_rollouts, vocab_size) — for each
        problem and rollout, the logits at latent step 1 projected to
        the vocabulary via the unembedding head. We do not need the
        actual sampling decision — only the distribution.

    Returns
    -------
    Scalar Python float: average natural-log entropy in nats.
    """
    if first_step_logits.dim() != 3:
        raise ValueError(
            f"expected (problems, rollouts, vocab); got "
            f"shape {tuple(first_step_logits.shape)}"
        )
    p = _stable_softmax(first_step_logits, dim=-1)
    log_p = torch.log(p.clamp_min(1e-10))
    per_rollout_entropy = -(p * log_p).sum(dim=-1)  # (problems, rollouts)
    return float(per_rollout_entropy.mean().item())


def halting_entropy(halt_steps: list[int] | torch.Tensor) -> float:
    """Empirical entropy of the halting-step distribution
    (proposal §2.7.4).

    Parameters
    ----------
    halt_steps:
        A 1-D iterable (or 1-D tensor) of integer halt-step indices,
        one per trajectory in the rollout group(s) being measured. We
        flatten across rollout groups; the §5.2.1 convention is to
        average across 256 rollout groups.

    Returns
    -------
    Scalar Python float in nats. Pre-registered alarm if this drops
    below 0.3 by cycle 2.
    """
    if isinstance(halt_steps, torch.Tensor):
        halt_steps = halt_steps.flatten().tolist()
    if not halt_steps:
        return 0.0
    import math

    counts = Counter(halt_steps)
    total = sum(counts.values())
    entropy = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            entropy -= p * math.log(p)
    return entropy


def post_block_ppl_ratio(
    post_block_logits: torch.Tensor,
    post_block_token_ids: torch.Tensor,
    *,
    base_ppl: float,
) -> float:
    """Ratio of per-token PPL of post-``</think>`` tokens to base PPL on
    the same continuation (architecture §2.3.1).

    Pre-registered alarm: ratio > 1.5 means the noise inside the latent
    block is corrupting downstream generation; the pilot diagnostic
    ladder fires (proposal §1.7).

    Parameters
    ----------
    post_block_logits:
        (T, vocab_size) — logits at each post-block token position.
    post_block_token_ids:
        (T,) — actual token ids that were emitted; we measure
        cross-entropy against these.
    base_ppl:
        The base model's PPL on the same continuation, computed on a
        clean (no-latent-noise) forward pass.
    """
    if post_block_logits.dim() != 2:
        raise ValueError(
            f"expected (T, vocab); got {tuple(post_block_logits.shape)}"
        )
    if post_block_token_ids.shape[0] != post_block_logits.shape[0]:
        raise ValueError("logits and token_ids must agree on length T")

    log_p = torch.log_softmax(post_block_logits.float(), dim=-1)
    nll = -log_p.gather(1, post_block_token_ids.long().unsqueeze(-1)).squeeze(-1)
    ppl = float(torch.exp(nll.mean()).item())
    return ppl / max(base_ppl, 1e-9)


def latent_diversity_alarm(
    cycle_to_entropy: dict[int, float],
    *,
    drop_threshold: float = 0.30,
) -> tuple[bool, str]:
    """Cycle-to-cycle latent first-step entropy alarm
    (proposal §2.7.3).

    Returns ``(alarm_fired, reason)``. ``alarm_fired = True`` when
    entropy at cycle ``c`` is more than ``drop_threshold`` below entropy
    at cycle ``c-1``; LDPT-SFT is mode-collapsing.
    """
    cycles_sorted = sorted(cycle_to_entropy.keys())
    for prev, curr in zip(cycles_sorted, cycles_sorted[1:]):
        e_prev = cycle_to_entropy[prev]
        e_curr = cycle_to_entropy[curr]
        if e_prev <= 0:
            continue
        rel_drop = (e_prev - e_curr) / e_prev
        if rel_drop > drop_threshold:
            return True, (
                f"latent first-step entropy dropped {rel_drop:.1%} from "
                f"cycle {prev} ({e_prev:.3f}) to cycle {curr} ({e_curr:.3f})"
            )
    return False, "no alarm"


def halting_short_circuit_alarm(
    cycle_to_entropy: dict[int, float],
    *,
    threshold_nats: float = 0.30,
    earliest_cycle: int = 2,
) -> tuple[bool, str]:
    """Halting-entropy alarm (proposal §2.7.4): fires if the halting
    entropy drops below ``threshold_nats`` by cycle ``earliest_cycle``.
    """
    for c, e in cycle_to_entropy.items():
        if c >= earliest_cycle and e < threshold_nats:
            return True, (
                f"halting entropy {e:.3f} nats < {threshold_nats:.2f} at "
                f"cycle {c} — short-circuit collapse"
            )
    return False, "no alarm"

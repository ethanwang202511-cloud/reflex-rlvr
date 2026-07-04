"""Evaluation primitives: pass@k from rollout records, NSR confidence,
SAE-feature novelty (math only here; the actual SAE encode happens in
the modal_app worker)."""

from reflex_rlvr.eval.pass_at_k import (
    bootstrap_pass_at_k_ci,
    pass_at_k_estimator,
    pass_at_k_per_problem,
)

__all__ = [
    "pass_at_k_estimator",
    "pass_at_k_per_problem",
    "bootstrap_pass_at_k_ci",
]

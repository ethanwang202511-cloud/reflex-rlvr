"""Latent register: noise schedule, halting head, soft-embedding
feedback (vLLM-side; stub here), entropy diagnostics.

The hot generation path lives in the vLLM fork; this module owns the
small, testable primitives that are shared between training and
inference and that we want to unit-test on Mac CPU.
"""

from reflex_rlvr.latent.cosine_anneal import cosine_anneal_noise
from reflex_rlvr.latent.diagnostics import (
    halting_entropy,
    latent_first_step_entropy,
    post_block_ppl_ratio,
)
from reflex_rlvr.latent.halt_head import HaltHead

__all__ = [
    "cosine_anneal_noise",
    "HaltHead",
    "latent_first_step_entropy",
    "halting_entropy",
    "post_block_ppl_ratio",
]

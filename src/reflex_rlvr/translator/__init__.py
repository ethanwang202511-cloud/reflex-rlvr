"""LDPT translator — stub.

The Modal-backed translator (LoRA-rank-64 adapter on Qwen2.5-7B,
trained on collected latent trajectories with verifier-only correctness
signal) lives in ``modal_app/``. This module exists for import safety.

Public API will be:

    from reflex_rlvr.translator import train_translator, translate_batch

…both of which dispatch to the Modal app at runtime.
"""

from __future__ import annotations


def train_translator(*args, **kwargs):  # noqa: D401, ANN002, ANN003
    raise NotImplementedError(
        "LoRA-translator training is GPU-bound; implement against modal_app/."
    )


def translate_batch(*args, **kwargs):  # noqa: D401, ANN002, ANN003
    raise NotImplementedError(
        "Latent-to-discrete translation is GPU-bound; implement against modal_app/."
    )

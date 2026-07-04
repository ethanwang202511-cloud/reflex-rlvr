"""Hard-set mining — stub.

The Modal-backed mining pipeline (vLLM batched inference + verifier
parallelism) lives in ``modal_app/``. This module exists so that
``reflex_rlvr.mining`` is import-safe on Mac CPU (the calling scripts
don't fail to import even before Modal is configured).

Public API will be:

    from reflex_rlvr.mining import build_hard_set, refresh_hard_set

…both of which dispatch to the Modal app at runtime.
"""

from __future__ import annotations


def build_hard_set(*args, **kwargs):  # noqa: D401, ANN002, ANN003
    raise NotImplementedError(
        "Hard-set mining is GPU-bound; implement against modal_app/ "
        "before calling. See architecture §5.1 stage 1."
    )


def refresh_hard_set(*args, **kwargs):  # noqa: D401, ANN002, ANN003
    raise NotImplementedError(
        "Hard-set refresh runs against the *current* base every 2 cycles. "
        "Implement against modal_app/ before calling."
    )

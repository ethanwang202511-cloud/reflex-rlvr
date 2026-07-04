"""LeanDojo-backed Lean 4 proof verifier.

This module is loaded *only inside the Modal verifier image* — the
``configs/modal.yaml`` ``images.verifier`` config installs elan +
Lean 4 + LeanDojo. On the Mac development host these imports raise
ImportError and ``lean_verifier.is_available()`` returns False, so
the router gracefully falls back to ``lean_not_installed``.

The Lean verifier is on the *critical path* only for the main 7B run
(15% Lean-mix in `H_K`, architecture §5.2). It is NOT used in the
Week-1 1.5B pilot (AIME math only) — so its absence on the Mac dev
host does not block the pilot launch.

Public surface (mirrors ``lean_verifier.verify_lean``):

    verify_lean_modal(proof_term, statement, *, timeout=30.0,
                      cache_key=None) -> LeanVerification
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache


@dataclass
class LeanVerification:
    accepted: bool
    reason: str
    runtime_ms: float | None = None
    proof_state: str | None = None


@lru_cache(maxsize=1)
def _get_lean_dojo():
    """Lazy import: only succeeds inside the verifier Modal image."""
    from lean_dojo import LeanGitRepo, Theorem, Dojo  # noqa: F401, PLC0415

    return (LeanGitRepo, Theorem, Dojo)


def _run_lean_kernel(proof_term: str, statement: str, *, timeout: float) -> tuple[bool, str]:
    """Run a single Lean kernel check.

    The exact LeanDojo invocation depends on the repo / theorem we
    target. For the REFLEX-RLVR setup we will pin a small Mathlib
    snapshot and treat each `(statement, proof_term)` pair as a
    fresh proof attempt against an in-memory dummy theorem. A full
    pinned-mathlib worker is engineering work to be completed before
    the main 7B Lean-mix runs (architecture §5.2).

    For the pilot (no Lean problems in the mix), this function is not
    called. We surface a clear NotImplementedError so a future Modal
    deploy that *does* include Lean can replace this stub with the
    pinned-mathlib worker without touching the router.
    """
    raise NotImplementedError(
        "Pinned-mathlib Lean worker not yet implemented. The Week-1 "
        "1.5B pilot does not use Lean (AIME math only). Build this "
        "worker before the 7B main run."
    )


def verify_lean_modal(
    proof_term: str,
    statement: str,
    *,
    timeout: float = 30.0,
    cache_key: str | None = None,
) -> LeanVerification:
    t0 = time.time()
    try:
        accepted, reason = _run_lean_kernel(proof_term, statement, timeout=timeout)
        runtime_ms = (time.time() - t0) * 1000.0
        return LeanVerification(
            accepted=accepted, reason=reason, runtime_ms=runtime_ms
        )
    except NotImplementedError as exc:
        return LeanVerification(
            accepted=False,
            reason=f"lean_worker_not_implemented:{exc}",
            runtime_ms=(time.time() - t0) * 1000.0,
        )
    except Exception as exc:  # noqa: BLE001
        return LeanVerification(
            accepted=False,
            reason=f"lean_kernel_error:{exc}",
            runtime_ms=(time.time() - t0) * 1000.0,
        )

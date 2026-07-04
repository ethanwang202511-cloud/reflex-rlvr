"""Lean 4 + LeanDojo verifier — STUB.

Lean is a heavy install (~1.2 GB elan toolchain). It is not installed
on the development Mac; this stub exists so the verifier router can
already dispatch to it, fall back gracefully, and the rest of the
pipeline can be tested without the Lean kernel hot.

When the Lean image is built on Modal (``configs/modal.yaml``,
``images.verifier``), this module's implementation is replaced with the
LeanDojo-backed verifier; the public ``verify_lean`` signature is
identical to the SymPy / code primitives.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LeanVerification:
    accepted: bool
    reason: str
    runtime_ms: float | None = None
    proof_state: str | None = None


def verify_lean(
    proof_term: str,
    statement: str,
    *,
    timeout: float = 30.0,
    cache_key: str | None = None,
) -> LeanVerification:
    """Verify a Lean 4 proof term against a goal statement.

    Inside the Modal verifier image, dispatches to the LeanDojo-backed
    implementation in ``_lean_modal.py``. On the Mac development host
    (no Lean kernel installed) raises ``LeanNotInstalled``, which the
    router catches and reports as ``lean_not_installed``.
    """
    if is_available():
        # Modal verifier image: LeanDojo + elan are installed; lazy-load
        # the worker so this module remains import-safe on Mac.
        from reflex_rlvr.verifier._lean_modal import verify_lean_modal

        result = verify_lean_modal(
            proof_term, statement, timeout=timeout, cache_key=cache_key
        )
        return LeanVerification(
            accepted=result.accepted,
            reason=result.reason,
            runtime_ms=result.runtime_ms,
            proof_state=result.proof_state,
        )

    raise LeanNotInstalled(
        "Lean 4 is not installed in this environment. The verifier "
        "router will fall back to a verifier-skipped status; this is "
        "expected behavior on the Mac development host."
    )


class LeanNotInstalled(RuntimeError):
    """Raised when Lean dispatch is attempted in an environment without
    the Lean kernel available. Caught by the verifier router."""


def is_available() -> bool:
    """Cheap probe used by the router to decide whether to dispatch."""
    import shutil

    return shutil.which("lean") is not None

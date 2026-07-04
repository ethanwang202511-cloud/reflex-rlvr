"""Top-level verifier dispatch.

A *Problem* is a dict with at minimum a ``domain`` field; the router
sends it to the appropriate primitive verifier and returns a unified
``VerifierResult``. Use :func:`verify` from outside this package — the
underlying primitives (sympy_verifier, code_verifier, lean_verifier)
are implementation detail.

The router is intentionally thin: dispatch + result normalization,
nothing more. Heavier infrastructure (parallel pools, caches, gVisor
sandboxes) lives in ``modal_app/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reflex_rlvr.verifier import code_verifier as cv
from reflex_rlvr.verifier import lean_verifier as lv
from reflex_rlvr.verifier import sympy_verifier as sv


@dataclass
class VerifierResult:
    accepted: bool
    domain: str
    reason: str
    extra: dict[str, Any] | None = None

    def __bool__(self) -> bool:  # convenience: ``if verify(...)``
        return self.accepted


def verify(problem: dict, candidate: str | dict) -> VerifierResult:
    """Verify *candidate* against *problem*.

    Parameters
    ----------
    problem:
        Dict with at minimum ``domain`` (one of ``math``, ``code``,
        ``arc``, ``lean``) and a domain-specific ``answer`` /
        ``test_cases`` / ``statement`` field.
    candidate:
        For ``math`` / ``arc`` / ``lean`` domains, a string of free-form
        completion text. For ``code``, the candidate is the source code
        — pass either a raw string or a dict ``{"code": str}``.
    """
    domain = problem.get("domain")
    if domain == "math":
        gt = problem["answer"]
        text = candidate if isinstance(candidate, str) else candidate.get("text", "")
        sub = sv.verify(text, gt)
        return VerifierResult(
            accepted=sub.accepted,
            domain="math",
            reason=sub.reason,
            extra={
                "parsed_candidate": sub.parsed_candidate,
                "canonical_diff": sub.canonical_diff,
            },
        )

    if domain == "arc":
        gt_grid = problem["answer_grid"]
        cand_grid = candidate if isinstance(candidate, list) else candidate.get("grid")
        accepted = cand_grid == gt_grid
        return VerifierResult(
            accepted=accepted,
            domain="arc",
            reason="ok" if accepted else "grid_mismatch",
        )

    if domain == "code":
        code = candidate if isinstance(candidate, str) else candidate.get("code", "")
        test_cases = problem.get("test_cases", [])
        sub = cv.verify_code(code, test_cases)
        return VerifierResult(
            accepted=sub.accepted,
            domain="code",
            reason=sub.reason,
            extra={
                "stdout": sub.stdout,
                "stderr": sub.stderr,
                "runtime_ms": sub.runtime_ms,
            },
        )

    if domain == "lean":
        proof = candidate if isinstance(candidate, str) else candidate.get("proof", "")
        try:
            sub = lv.verify_lean(proof, problem.get("statement", ""))
            return VerifierResult(
                accepted=sub.accepted,
                domain="lean",
                reason=sub.reason,
                extra={"runtime_ms": sub.runtime_ms},
            )
        except lv.LeanNotInstalled:
            # Mac development path: report verifier-skipped, treat as
            # not-accepted so the calling code can choose to drop the
            # trajectory or route around it.
            return VerifierResult(
                accepted=False,
                domain="lean",
                reason="lean_not_installed",
            )

    return VerifierResult(
        accepted=False, domain=str(domain), reason="unknown_domain"
    )

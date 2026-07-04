"""Verifier sandbox: SymPy + Lean 4 + sandboxed code execution.

The verifier is the only correctness signal in REFLEX-RLVR — there is
no neural teacher. Bugs here poison the RL gradient, so this module
is heavily unit-tested before any GPU job is launched.

Public entry point: ``verify(problem, candidate)`` from ``router``.
"""

from reflex_rlvr.verifier.router import VerifierResult, verify

__all__ = ["verify", "VerifierResult"]

"""SymPy-based verifier for math-answer problems (AIME, HMMT, MATH).

Parses a candidate's final ``\\boxed{...}`` answer (with several common
fallbacks), canonicalizes via SymPy, and compares to the ground truth.
Target latency ≤ 5 ms / check on Mac CPU.

This module is import-safe with no heavy dependencies beyond SymPy
itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import sympy as sp
from sympy.parsing.latex import parse_latex
from sympy.parsing.sympy_parser import parse_expr

# Patterns, evaluated in order. The first one that yields a parsable
# expression wins. The tail patterns catch common AoPS / AIME formats.
_BOXED_PATTERNS = [
    re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"),
    re.compile(r"\\boxed\s+([^\s$]+)"),
    re.compile(r"\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"),
    re.compile(r"answer\s*[:=]\s*([^\n.]+)", re.IGNORECASE),
    re.compile(r"final answer\s*[:=]?\s*([^\n.]+)", re.IGNORECASE),
    re.compile(r"\\\((.+?)\\\)\s*$", re.DOTALL),
    re.compile(r"\$\$?(.+?)\$\$?\s*$", re.DOTALL),
]


@dataclass
class SympyVerification:
    accepted: bool
    reason: str
    parsed_candidate: str | None = None
    canonical_diff: str | None = None


def _resolve_named_constants(expr: sp.Expr) -> sp.Expr:
    """Substitute well-known free symbols (``pi``, ``e``, ``i``, ``oo``)
    introduced by ``parse_latex`` / ``parse_expr`` with their canonical
    SymPy constant counterparts, so ``float(expr)`` works.
    """
    subs = {}
    for atom in expr.free_symbols:
        name = atom.name
        if name == "pi":
            subs[atom] = sp.pi
        elif name == "e":
            subs[atom] = sp.E
        elif name in ("i", "I"):
            subs[atom] = sp.I
        elif name in ("oo", "infty", "infinity"):
            subs[atom] = sp.oo
    return expr.subs(subs) if subs else expr


def _strip_outer_dollars(s: str) -> str:
    s = s.strip()
    if s.startswith("$$") and s.endswith("$$"):
        return s[2:-2].strip()
    if s.startswith("$") and s.endswith("$"):
        return s[1:-1].strip()
    return s


def extract_answer(text: str) -> str | None:
    """Pull the candidate answer string out of a free-form completion.

    Returns the inner expression as a string, or None if no pattern
    matches. The caller is expected to feed the result to
    :func:`canonicalize`.
    """
    if not text:
        return None
    # Walk the patterns and return the LAST match for each — competition
    # solutions often state intermediate boxed quantities and only the
    # final one is the answer.
    for pat in _BOXED_PATTERNS:
        matches = pat.findall(text)
        if matches:
            return _strip_outer_dollars(matches[-1])
    return None


def canonicalize(s: str) -> sp.Expr | None:
    """Canonicalize a candidate-answer string into a SymPy expression.

    Returns None on parse failure. We try LaTeX first, then sympy
    parser, then a numeric fallback. Intentionally permissive — the
    downstream comparison is exact / structural, so a permissive
    parse front-end does not cause false positives.
    """
    s = (s or "").strip().rstrip(".,;").strip()
    if not s:
        return None

    # Cleanups that are safe across parsers.
    s_clean = (
        s.replace("\\\\", "\\")
        .replace("\\!", "")
        .replace("\\,", "")
        .replace("\\;", "")
        .replace("\\:", "")
        .replace("\\ ", " ")
        .replace("\\left", "")
        .replace("\\right", "")
        .replace("\\dfrac", "\\frac")
        .replace("\\tfrac", "\\frac")
        .replace("^{\\circ}", "")
        .replace("^\\circ", "")
        .replace("\\degree", "")
        .replace("^o", "")
    )

    # Strip trailing units that SymPy can't parse (degrees, percent, etc.)
    s_clean = re.sub(r"\\text\{[^}]*\}", "", s_clean)
    s_clean = re.sub(r"\s*\\?%\s*$", "", s_clean)

    # 1. Try the LaTeX parser.
    if "\\" in s_clean or "{" in s_clean or "frac" in s_clean:
        try:
            parsed = parse_latex(s_clean)
            return sp.simplify(_resolve_named_constants(parsed))
        except Exception:  # noqa: BLE001 — parse_latex raises a wide variety
            pass

    # 2. Try the sympy text parser (handles "1/2", "sqrt(2)", "pi/3").
    try:
        parsed = parse_expr(s_clean.replace("^", "**"))
        return sp.simplify(_resolve_named_constants(parsed))
    except Exception:  # noqa: BLE001
        pass

    # 3. Numeric fallback — strip whitespace and try float / Rational.
    try:
        return sp.Rational(s_clean.replace(",", "").replace(" ", ""))
    except Exception:  # noqa: BLE001
        pass

    return None


def equal(a: sp.Expr | None, b: sp.Expr | None, *, tol: float = 1e-9) -> bool:
    """Structural and numeric equality, in that order."""
    if a is None or b is None:
        return False
    try:
        # Exact symbolic.
        if sp.simplify(a - b) == 0:
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        af = float(a)
        bf = float(b)
        return abs(af - bf) < tol
    except Exception:  # noqa: BLE001
        return False


def verify(
    candidate_text: str,
    ground_truth: str | int | float,
) -> SympyVerification:
    """Verify a candidate completion against a ground-truth answer.

    Parameters
    ----------
    candidate_text:
        Free-form completion text from the model.
    ground_truth:
        The canonical answer. Strings are SymPy-parsed; numerics are
        wrapped in Rational where possible.
    """
    extracted = extract_answer(candidate_text)
    if extracted is None:
        return SympyVerification(accepted=False, reason="no_boxed_answer_found")

    candidate = canonicalize(extracted)
    if candidate is None:
        return SympyVerification(
            accepted=False,
            reason="candidate_parse_failed",
            parsed_candidate=extracted,
        )

    if isinstance(ground_truth, (int, float)):
        gt: sp.Expr = sp.nsimplify(ground_truth, rational=True)
    else:
        gt_parsed = canonicalize(str(ground_truth))
        if gt_parsed is None:
            return SympyVerification(
                accepted=False,
                reason="ground_truth_parse_failed",
                parsed_candidate=extracted,
            )
        gt = gt_parsed

    if equal(candidate, gt):
        return SympyVerification(
            accepted=True, reason="ok", parsed_candidate=str(candidate)
        )
    return SympyVerification(
        accepted=False,
        reason="not_equal",
        parsed_candidate=str(candidate),
        canonical_diff=str(sp.simplify(candidate - gt)),
    )

"""Tests for the verifier router — domain dispatch + result
normalization."""

from __future__ import annotations

from reflex_rlvr.verifier import verify


class TestMathDomain:
    def test_correct_math_answer(self) -> None:
        problem = {"domain": "math", "answer": 42}
        candidate = r"After computation, \boxed{42}."
        result = verify(problem, candidate)
        assert result.accepted
        assert result.domain == "math"

    def test_wrong_math_answer(self) -> None:
        problem = {"domain": "math", "answer": 42}
        candidate = r"\boxed{43}"
        result = verify(problem, candidate)
        assert not result.accepted

    def test_truthiness_short_circuits(self) -> None:
        problem = {"domain": "math", "answer": 1}
        result = verify(problem, r"\boxed{1}")
        assert bool(result) is True


class TestCodeDomain:
    def test_correct_code(self) -> None:
        problem = {
            "domain": "code",
            "test_cases": [
                {"input": "5\n", "expected": "25"},
                {"input": "3\n", "expected": "9"},
            ],
        }
        code = "import sys\nx = int(sys.stdin.read())\nprint(x*x)"
        result = verify(problem, code)
        assert result.accepted, result.reason


class TestArcDomain:
    def test_correct_grid(self) -> None:
        problem = {"domain": "arc", "answer_grid": [[1, 0], [0, 1]]}
        result = verify(problem, [[1, 0], [0, 1]])
        assert result.accepted

    def test_grid_mismatch(self) -> None:
        problem = {"domain": "arc", "answer_grid": [[1, 0], [0, 1]]}
        result = verify(problem, [[0, 1], [1, 0]])
        assert not result.accepted
        assert result.reason == "grid_mismatch"


class TestLeanDomainStub:
    def test_falls_back_to_not_installed(self) -> None:
        problem = {"domain": "lean", "statement": "True"}
        result = verify(problem, "trivial")
        # On a Mac dev machine without lean installed, the router
        # gracefully reports lean_not_installed without crashing.
        assert not result.accepted
        assert result.reason == "lean_not_installed"


class TestUnknownDomain:
    def test_unknown_returns_not_accepted(self) -> None:
        problem = {"domain": "physics"}
        result = verify(problem, "F = ma")
        assert not result.accepted
        assert result.reason == "unknown_domain"

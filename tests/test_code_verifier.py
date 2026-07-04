"""Tests for the sandboxed code verifier."""

from __future__ import annotations

from reflex_rlvr.verifier.code_verifier import (
    run_with_stdin,
    verify_code,
    verify_code_json_io,
)


class TestRunWithStdin:
    def test_hello_world(self) -> None:
        result = run_with_stdin("print('hello')")
        assert result.accepted
        assert result.stdout.strip() == "hello"

    def test_stdin_passthrough(self) -> None:
        code = "import sys\nprint(sys.stdin.read().strip()[::-1])"
        result = run_with_stdin(code, stdin="abcdef")
        assert result.accepted
        assert result.stdout.strip() == "fedcba"

    def test_timeout_kills_long_loop(self) -> None:
        code = "while True: pass"
        result = run_with_stdin(code, timeout=0.5)
        assert not result.accepted
        assert result.reason == "timeout"

    def test_runtime_error_caught(self) -> None:
        code = "raise RuntimeError('boom')"
        result = run_with_stdin(code)
        assert not result.accepted
        assert result.exit_code != 0

    def test_blocked_module_socket(self) -> None:
        code = "import socket\nprint(socket.gethostname())"
        result = run_with_stdin(code)
        assert not result.accepted
        assert "blocked by sandbox" in (result.stderr or "")


class TestVerifyCode:
    def test_simple_addition_problem(self) -> None:
        code = (
            "import sys\n"
            "a, b = map(int, sys.stdin.read().split())\n"
            "print(a + b)"
        )
        cases = [
            {"input": "1 2\n", "expected": "3"},
            {"input": "10 20\n", "expected": "30"},
            {"input": "-5 5\n", "expected": "0"},
        ]
        result = verify_code(code, cases)
        assert result.accepted, result.reason

    def test_one_test_fails(self) -> None:
        code = "import sys\nprint(int(sys.stdin.read().strip()) * 2)"
        cases = [
            {"input": "3\n", "expected": "6"},
            {"input": "4\n", "expected": "9"},  # wrong on purpose
        ]
        result = verify_code(code, cases)
        assert not result.accepted
        assert "wrong_answer" in result.reason


class TestJSONIOVerifier:
    def test_solve_function_json_io(self) -> None:
        code = "def solve(x):\n    return x * x"
        result = verify_code_json_io(code, [3, 4, 5], [9, 16, 25])
        assert result.accepted

    def test_solve_function_wrong(self) -> None:
        code = "def solve(x):\n    return x"
        result = verify_code_json_io(code, [3, 4], [9, 16])
        assert not result.accepted

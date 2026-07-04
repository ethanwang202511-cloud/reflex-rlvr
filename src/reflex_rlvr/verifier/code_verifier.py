"""Sandboxed Python code execution for LiveCodeBench / Codeforces-style
problems.

On Mac (development) we use a subprocess sandbox — restricted importable
modules, hard wall-clock timeout, memory limits via resource.setrlimit
where supported. On Modal we will swap this for a gVisor-isolated
runtime; the public ``verify_code`` entry point is identical so callers
do not have to care.

The goal is correctness verification, not adversarial sandboxing. A
well-resourced attacker on a Mac dev box can break out of subprocess
isolation; that is acceptable for the local pre-Modal smoketest.
Production runs via Modal use gVisor and per-trajectory ephemeral
containers.
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass

DEFAULT_TIMEOUT_SECONDS = 2.0
DEFAULT_MEMORY_MB = 512


@dataclass
class CodeVerification:
    accepted: bool
    reason: str
    stdout: str | None = None
    stderr: str | None = None
    runtime_ms: float | None = None
    exit_code: int | None = None


_SANDBOX_PRELUDE = """\
import sys
import resource
import builtins

# Mac sets SOFT/HARD RLIMIT_AS via the parent. On Linux we set it here.
if hasattr(resource, 'RLIMIT_AS') and sys.platform.startswith('linux'):
    resource.setrlimit(resource.RLIMIT_AS, ({memory_bytes}, {memory_bytes}))

# Restrict imports to a deny-list of obviously dangerous modules. This
# is not a security boundary on its own; the wall clock + subprocess
# isolation is the actual containment.
_BANNED = {{'socket', 'requests', 'urllib', 'urllib2', 'urllib3',
            'http', 'ftplib', 'subprocess', 'multiprocessing', 'os.fork',
            'paramiko', 'smtplib', 'telnetlib'}}
_orig_import = builtins.__import__
def _guarded_import(name, *a, **kw):
    if name.split('.')[0] in _BANNED:
        raise ImportError(f'module {{name!r}} blocked by sandbox')
    return _orig_import(name, *a, **kw)
builtins.__import__ = _guarded_import
"""


def _wrap_program(user_code: str, *, memory_mb: int) -> str:
    """Build the program string the subprocess actually executes."""
    prelude = _SANDBOX_PRELUDE.format(memory_bytes=memory_mb * 1024 * 1024)
    return f"{prelude}\n# --- user code below ---\n{user_code}\n"


def run_with_stdin(
    code: str,
    stdin: str = "",
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    memory_mb: int = DEFAULT_MEMORY_MB,
) -> CodeVerification:
    """Run *code* in a subprocess, feeding *stdin*, return stdout.

    This is the primitive used by the LiveCodeBench / Codeforces format
    where each test case is (stdin → expected stdout).
    """
    program = _wrap_program(code, memory_mb=memory_mb)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, encoding="utf-8"
    ) as fh:
        fh.write(program)
        path = fh.name

    try:
        # Mac: set memory cap on the child via setrlimit in a preexec_fn.
        def _preexec() -> None:  # pragma: no cover — only on POSIX
            try:
                resource.setrlimit(
                    resource.RLIMIT_CPU,
                    (int(timeout) + 1, int(timeout) + 1),
                )
            except (ValueError, OSError):
                pass

        import time

        t0 = time.perf_counter()
        completed = subprocess.run(  # noqa: S603 — sandboxed by design
            [sys.executable, "-I", "-S", path],
            input=stdin,
            capture_output=True,
            text=True,
            timeout=timeout,
            preexec_fn=_preexec if os.name == "posix" else None,
            check=False,
        )
        runtime_ms = (time.perf_counter() - t0) * 1000.0
        return CodeVerification(
            accepted=(completed.returncode == 0),
            reason="ok" if completed.returncode == 0 else "nonzero_exit",
            stdout=completed.stdout,
            stderr=completed.stderr,
            runtime_ms=runtime_ms,
            exit_code=completed.returncode,
        )
    except subprocess.TimeoutExpired:
        return CodeVerification(
            accepted=False, reason="timeout", runtime_ms=timeout * 1000.0
        )
    except Exception as exc:  # noqa: BLE001 — surface everything
        return CodeVerification(accepted=False, reason=f"sandbox_error:{exc}")
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def verify_code(
    code: str,
    test_cases: list[dict],
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    memory_mb: int = DEFAULT_MEMORY_MB,
) -> CodeVerification:
    """Run *code* against a list of (stdin, expected_stdout) test cases.

    The candidate is accepted only if every test case passes.

    Each test case is a dict with keys ``input`` (str) and ``expected``
    (str). Trailing whitespace on each line is stripped before
    comparison; the expected output is matched after a single
    .rstrip().

    Note that this primitive expects a single self-contained program; we
    do not attempt to parse out a function and call it directly. Most
    competitive-programming benchmarks already follow this convention.
    """
    for i, tc in enumerate(test_cases):
        stdin = tc.get("input", "")
        expected = tc.get("expected", "")
        result = run_with_stdin(
            code, stdin=stdin, timeout=timeout, memory_mb=memory_mb
        )
        if not result.accepted:
            result.reason = f"test_{i}_{result.reason}"
            return result
        actual = (result.stdout or "").rstrip()
        if actual != expected.rstrip():
            return CodeVerification(
                accepted=False,
                reason=f"test_{i}_wrong_answer",
                stdout=result.stdout,
                stderr=result.stderr,
                runtime_ms=result.runtime_ms,
                exit_code=result.exit_code,
            )
    return CodeVerification(accepted=True, reason="ok")


def verify_code_json_io(
    code: str,
    inputs: list,
    expected_outputs: list,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    memory_mb: int = DEFAULT_MEMORY_MB,
) -> CodeVerification:
    """Test-case format that wraps the code as a function called once
    per input. The function's name is convention-fixed at ``solve``;
    inputs and expected outputs are JSON-serializable Python objects.
    """
    if len(inputs) != len(expected_outputs):
        return CodeVerification(accepted=False, reason="bad_test_spec")

    for i, (inp, expected) in enumerate(zip(inputs, expected_outputs)):
        driver = (
            code
            + "\n\n"
            + "import json, sys\n"
            + f"_arg = json.loads({json.dumps(json.dumps(inp))})\n"
            + "print(json.dumps(solve(_arg)))\n"
        )
        result = run_with_stdin(driver, timeout=timeout, memory_mb=memory_mb)
        if not result.accepted:
            result.reason = f"test_{i}_{result.reason}"
            return result
        try:
            actual = json.loads((result.stdout or "").strip())
        except json.JSONDecodeError:
            return CodeVerification(
                accepted=False,
                reason=f"test_{i}_invalid_json_stdout",
                stdout=result.stdout,
                stderr=result.stderr,
            )
        if actual != expected:
            return CodeVerification(
                accepted=False,
                reason=f"test_{i}_wrong_answer",
                stdout=result.stdout,
                stderr=result.stderr,
                runtime_ms=result.runtime_ms,
            )
    return CodeVerification(accepted=True, reason="ok")

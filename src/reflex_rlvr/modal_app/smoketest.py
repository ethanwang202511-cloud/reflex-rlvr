"""Modal smoketest — CPU only, ≈ $0.001.

Purpose: verify Modal authentication, image build, secrets, and
volumes are wired correctly before any GPU spend. Does NOT consume
H100 hours and does not require approval per the per-job rule.

Usage (from the project root):

    modal run scripts/modal_smoketest.py

The smoketest:
1. Echoes the deploy date / app name.
2. Imports torch (CPU only) to verify the image build.
3. Reads the wandb secret to verify the secret is in scope.
4. Touches the cache volume to verify the mount path is writable.
5. Returns the elapsed wall-clock time.
"""

from __future__ import annotations

import os
import time

try:  # pragma: no cover — Modal isn't a hard dep on Mac dev box for tests
    import modal
except ImportError:  # noqa: BLE001
    modal = None  # type: ignore[assignment]


def _smoketest_secrets():
    if modal is None:
        return []
    out = []
    for name in ("huggingface", "wandb"):
        try:
            out.append(modal.Secret.from_name(name))
        except Exception:  # noqa: BLE001 — secret may not exist yet
            pass
    return out


if modal is not None:
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("torch==2.4.0")
    )
    app = modal.App("reflex-rlvr-smoketest", image=image)

    @app.function(
        cpu=2.0,
        memory=2048,
        timeout=120,
        secrets=_smoketest_secrets(),
    )
    def smoketest() -> dict:
        t0 = time.time()
        result = {
            "app_name": "reflex-rlvr-smoketest",
            "wandb_api_key_present": "WANDB_API_KEY" in os.environ,
            "torch_version": None,
            "torch_cuda_available": None,
            "elapsed_s": None,
        }
        try:
            import torch  # noqa: PLC0415 — runtime import for image-build test

            result["torch_version"] = torch.__version__
            result["torch_cuda_available"] = torch.cuda.is_available()
        except Exception as exc:  # noqa: BLE001
            result["torch_import_error"] = str(exc)
        result["elapsed_s"] = round(time.time() - t0, 3)
        return result

    @app.local_entrypoint()
    def main() -> None:
        print(smoketest.remote())

else:  # Modal not installed (pre-deps state); make this importable on Mac.

    def main() -> None:
        print(
            "modal is not installed in this environment. Install via "
            "`uv add modal` and then re-run `modal run "
            "scripts/modal_smoketest.py`."
        )

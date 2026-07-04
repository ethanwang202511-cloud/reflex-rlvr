"""Common Modal-app primitives shared across mining / premise-test /
RL functions. Image build, volume mounts, secret refs, GPU profiles.

Single source of truth: ``configs/modal.yaml``. This module reads that
YAML at import time and re-exports the resulting Modal objects.
"""

from __future__ import annotations

from pathlib import Path

import yaml

try:  # Modal isn't a hard dep on Mac dev box for unit tests.
    import modal
except ImportError:  # noqa: BLE001
    modal = None  # type: ignore[assignment]


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs" / "modal.yaml"


def load_modal_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


_CFG = load_modal_config()


def _build_image(*, with_lean: bool = False):
    """Construct the Modal image per ``configs/modal.yaml``.

    The default image (no Lean) is enough for math + code + ARC. The
    ``with_lean=True`` image installs elan + Lean 4 + LeanDojo, used
    only by the verifier worker on the main 7B run.
    """
    if modal is None:
        return None
    spec = _CFG["image"]
    img = (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install(*spec["apt_packages"])
        .pip_install(*spec["pip_packages"])
        # Ship the local `reflex_rlvr` package into the worker so the
        # GPU function can `import reflex_rlvr.*` without re-pip-
        # installing inside the image. `add_local_python_source` is
        # the modal-recommended path for editable workspaces.
        .add_local_python_source("reflex_rlvr")
        # The runtime container re-imports `common.py`, which calls
        # `load_modal_config()` reading `configs/modal.yaml`. Inside
        # the container `__file__` resolves so that REPO_ROOT == "/",
        # so the config file must land at `/configs/modal.yaml`.
        .add_local_file(str(CONFIG_PATH), "/configs/modal.yaml")
    )
    if with_lean:
        img = img.run_commands(
            "curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh "
            "| sh -s -- -y",
            "echo 'export PATH=$HOME/.elan/bin:$PATH' >> /root/.bashrc",
        ).pip_install("lean-dojo>=2.2.0")
    return img


# Public Modal objects (None on Mac dev where modal isn't installed).
APP_NAME = _CFG.get("app_name", "reflex-rlvr")
IMAGE = _build_image(with_lean=False)
IMAGE_VERIFIER = _build_image(with_lean=True)


def _get_volume(name: str):
    if modal is None:
        return None
    return modal.Volume.from_name(name, create_if_missing=False)


VOLUME_CACHE = _get_volume("reflex-rlvr-cache")
VOLUME_DATA = _get_volume("reflex-rlvr-data")
VOLUME_CHECKPOINTS = _get_volume("reflex-rlvr-checkpoints")

VOLUMES = (
    {
        "/cache": VOLUME_CACHE,
        "/data": VOLUME_DATA,
        "/checkpoints": VOLUME_CHECKPOINTS,
    }
    if modal is not None
    else {}
)


def _get_secret(name: str):
    if modal is None:
        return None
    return modal.Secret.from_name(name)


# `huggingface` already exists in the user's Modal account.
# `wandb` is optional; create with
#   `modal secret create wandb WANDB_API_KEY=<key>`
SECRET_HF = _get_secret("huggingface")


def get_secrets(*, include_wandb: bool = True):
    """Return the list of Modal secrets to attach to a function.

    `huggingface` is always included. `wandb` is included when
    `include_wandb=True` AND the secret exists; if it doesn't, we
    silently drop it so functions can still launch without wandb
    configured.
    """
    if modal is None:
        return []
    out = [SECRET_HF]
    if include_wandb:
        try:
            out.append(modal.Secret.from_name("wandb"))
        except Exception:  # noqa: BLE001 — secret may not exist yet
            pass
    return out


# GPU profiles, by tag from ``configs/modal.yaml``.
GPU_PROFILES = _CFG["gpu_profiles"]


def gpu_spec(profile: str) -> str | None:
    """Return a Modal-friendly GPU spec like ``"H100:4"`` or None for CPU."""
    p = GPU_PROFILES[profile]
    if p["gpu"] is None:
        return None
    return f"{p['gpu']}:{p['n_gpus']}"


# Cost model — used by every function to print expected H100·hr cost
# *before* launch. Anchored to RunPod / Lambda spot rates 2026-Q1.
RATES_USD_PER_HOUR = {
    "H100": 2.39,
    "A100-80GB": 1.19,
    "A10G": 0.65,
    "T4": 0.15,
    None: 0.0,
}


def estimate_cost(gpu: str | None, n_gpus: int, hours: float) -> float:
    rate = RATES_USD_PER_HOUR.get(gpu, 0.0)
    return round(rate * n_gpus * hours, 2)

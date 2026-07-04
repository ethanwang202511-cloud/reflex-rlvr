"""Per-token surprise mechanistic probe (discrimination_v2 §F)."""
from reflex_rlvr.modal_app.discrimination_v2 import app, main_surprise as main  # noqa: F401
if __name__ == "__main__":
    raise SystemExit("Run via `modal run scripts/run_per_token_surprise.py ...`, not directly.")

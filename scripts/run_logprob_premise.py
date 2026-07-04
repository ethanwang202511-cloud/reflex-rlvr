"""Logprob-based premise test (paper §7 limitation #1 follow-up)."""
from reflex_rlvr.modal_app.premise_test import app, main_logprob as main  # noqa: F401
if __name__ == "__main__":
    raise SystemExit("Run via `modal run scripts/run_logprob_premise.py ...`, not directly.")

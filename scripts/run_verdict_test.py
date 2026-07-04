"""YES/NO verdict grader logit test (discrimination_v2 §C)."""
from reflex_rlvr.modal_app.discrimination_v2 import app, main_verdict as main  # noqa: F401
if __name__ == "__main__":
    raise SystemExit("Run via `modal run scripts/run_verdict_test.py ...`, not directly.")

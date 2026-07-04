"""Run the verdict probe on a base model merged with a SFT'd LoRA adapter."""
from reflex_rlvr.modal_app.sft import app, main_verdict_sft as main  # noqa: F401

if __name__ == "__main__":
    raise SystemExit(
        "Run via `modal run scripts/run_verdict_test_sft.py ...`, not directly."
    )

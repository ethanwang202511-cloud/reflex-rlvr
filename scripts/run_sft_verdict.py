"""SFT a LoRA adapter for verdict recovery on a base LLM.

Run via:

    modal run scripts/run_sft_verdict.py --train-jsonl data/sft/verdict_train.jsonl

The dataset is built by ``scripts/build_sft_dataset.py``.
"""
from reflex_rlvr.modal_app.sft import app, main_sft as main  # noqa: F401

if __name__ == "__main__":
    raise SystemExit(
        "Run via `modal run scripts/run_sft_verdict.py ...`, not directly."
    )

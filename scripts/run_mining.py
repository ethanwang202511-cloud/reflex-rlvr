"""Hard-set mining entrypoint — pass@k sweep on a candidate problem set.

Usage:

    # Pilot mining: 100 AIME 2018-2023 problems, pass@1024 on Qwen2.5-1.5B
    modal run scripts/run_mining.py \\
        --problems-jsonl data/aime/h_k_pilot.jsonl \\
        --k 1024 --model-id Qwen/Qwen2.5-1.5B

Cost estimate (Qwen2.5-1.5B, 4×H100, 100 problems × pass@1024):
    ≈ $4–8 wall-clock; vLLM batched, completes in ~5 min.

This entrypoint is a thin wrapper over
``reflex_rlvr.modal_app.mining``; the actual GPU function and cost
model live there.
"""

from reflex_rlvr.modal_app.mining import app, main  # noqa: F401

# Re-export `main` and `app` so `modal run scripts/run_mining.py` works.
if __name__ == "__main__":
    raise SystemExit(
        "Run via `modal run scripts/run_mining.py ...`, not directly with python."
    )

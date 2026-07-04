"""Premise test (Week-1 pilot gate (a), proposal §1.7).

Usage:

    # AIME 2018-2023 set:
    modal run scripts/run_premise_test.py \\
        --problems-jsonl data/aime/h_k_pilot.jsonl \\
        --n-samples 8

    # Synthetic memorization control:
    modal run scripts/run_premise_test.py \\
        --problems-jsonl data/aime/synthetic_pilot.jsonl \\
        --save-to /cache/premise/premise_test_synthetic.jsonl

The aggregated decision (60% pass with paired sign test α=0.05) is
made in ``scripts/decide_gate_a.py`` after both runs complete.

Cost estimate (Qwen2.5-1.5B, 4×H100, 100 problems × 4 conditions × 8 samples):
    ≈ $0.10 — negligible against the $30 premise-test budget line.
"""

from reflex_rlvr.modal_app.premise_test import app, main  # noqa: F401

if __name__ == "__main__":
    raise SystemExit(
        "Run via `modal run scripts/run_premise_test.py ...`, not directly with python."
    )

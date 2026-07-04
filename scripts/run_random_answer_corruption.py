"""Round-7 reviewer-defense probe: Probe-1 random-answer corruption control.

Swaps the final \\boxed{N_oracle} with an in-distribution AIME-style random
answer drawn from other problems' ground-truth answers. Chain content
unchanged. Isolates "real chain reading" from "out-of-distribution
token-frequency artifact" as the source of the σ_disc signal in
consistent_corruption.
"""
from reflex_rlvr.modal_app.discrimination_v2 import app, main_random_answer as main  # noqa: F401
if __name__ == "__main__":
    raise SystemExit("Run via `modal run scripts/run_random_answer_corruption.py ...`, not directly.")

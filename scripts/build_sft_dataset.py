"""Build the SFT training set (Phase 2 of Round 8) — runs locally, no Modal.

Outputs ``data/sft/verdict_train.jsonl`` with one (problem, solution, label)
row per oracle/corrupted variant. Uses NuminaMath AMC/AIME problems
(disjoint from the 51-problem eval set).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python scripts/build_sft_dataset.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reflex_rlvr.modal_app.sft import build_sft_dataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--numina-jsonl",
        default="data/aime/numina_amc_aime.jsonl",
    )
    p.add_argument(
        "--out-jsonl",
        default="data/sft/verdict_train.jsonl",
    )
    p.add_argument("--n-problems", type=int, default=500)
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    summary = build_sft_dataset(
        args.numina_jsonl,
        args.out_jsonl,
        n_problems=args.n_problems,
        seed=args.seed,
    )
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

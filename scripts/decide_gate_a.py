"""Aggregate the four-condition premise-test outputs into the
pre-registered gate (a) decision (proposal §1.7).

Usage:

    python scripts/decide_gate_a.py \\
        --aime-jsonl results/pilot/premise_test_1p5b.jsonl \\
        --synthetic-jsonl results/pilot/premise_test_synthetic.jsonl

Decision rule (locked):
    mean(p_disc_oracle) ≥ 0.5
    AND mean(p_disc_oracle) - mean(p_disc_corrupted) ≥ 0.2
    on at least 60% of problems
    paired sign test α=0.05.

Memorization-control corollary: the same gate must also pass on
the synthetic set, otherwise the AIME result might be memorization-
contaminated.

This script is *local* CPU only — it consumes the per-problem JSONL
files written by the Modal premise_test function and does the
statistical aggregation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _per_problem_pass(rows: list[dict], threshold_p_disc: float = 0.5,
                       threshold_delta: float = 0.2) -> list[dict]:
    out = []
    for r in rows:
        oracle = r.get("p_disc_oracle") or 0.0
        corrupted = r.get("p_disc_corrupted") or 0.0
        delta = oracle - corrupted
        out.append(
            {
                "id": r.get("id"),
                "p_disc_oracle": oracle,
                "p_disc_corrupted": corrupted,
                "delta": delta,
                "passes_gate": oracle >= threshold_p_disc and delta >= threshold_delta,
            }
        )
    return out


def _paired_sign_test(rows: list[dict], alpha: float = 0.05) -> dict:
    """Paired sign test on `(oracle - corrupted) > 0` per problem.

    H0: median delta = 0 (no asymmetry).
    H1: median delta > 0 (oracle > corrupted).

    Uses scipy.stats.binomtest if available; falls back to an exact
    binomial p-value calculation otherwise.
    """
    n_pos = sum(1 for r in rows if r["delta"] > 0)
    n_neg = sum(1 for r in rows if r["delta"] < 0)
    n_eff = n_pos + n_neg  # tied (delta = 0) excluded
    if n_eff == 0:
        return {"n_pos": 0, "n_neg": 0, "n_eff": 0, "p_value": 1.0,
                "reject_h0": False}
    try:
        from scipy.stats import binomtest

        result = binomtest(n_pos, n_eff, p=0.5, alternative="greater")
        p_value = float(result.pvalue)
    except ImportError:
        # Exact one-sided binomial using math.comb
        import math

        p_value = sum(
            math.comb(n_eff, k) for k in range(n_pos, n_eff + 1)
        ) / (2 ** n_eff)
    return {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_eff": n_eff,
        "p_value": float(p_value),
        "reject_h0": p_value < alpha,
    }


def aggregate(jsonl_path: Path, *, label: str) -> dict:
    rows = _load_jsonl(jsonl_path)
    n = len(rows)
    per = _per_problem_pass(rows)
    n_passing = sum(int(r["passes_gate"]) for r in per)
    pass_pct = n_passing / max(1, n)
    sign = _paired_sign_test(per)
    mean_oracle = sum(r["p_disc_oracle"] for r in per) / max(1, n)
    mean_corrupted = sum(r["p_disc_corrupted"] for r in per) / max(1, n)

    return {
        "label": label,
        "source": str(jsonl_path),
        "n_problems": n,
        "n_passing_per_problem": n_passing,
        "pct_passing_per_problem": round(pass_pct, 4),
        "mean_p_disc_oracle": round(mean_oracle, 4),
        "mean_p_disc_corrupted": round(mean_corrupted, 4),
        "delta_means": round(mean_oracle - mean_corrupted, 4),
        "paired_sign_test": sign,
        "gate_a_per_problem_60pct": pass_pct >= 0.6,
        "gate_a_sign_test_alpha_05": sign["reject_h0"],
        "gate_a_means_thresholds": (
            mean_oracle >= 0.5 and (mean_oracle - mean_corrupted) >= 0.2
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Decide gate (a) of the pilot.")
    parser.add_argument("--aime-jsonl", type=Path, required=True)
    parser.add_argument("--synthetic-jsonl", type=Path, default=None,
                        help="Optional synthetic memorization-control JSONL.")
    parser.add_argument("--out", type=Path, default=
                        REPO_ROOT / "results" / "pilot" / "gate_a_decision.json")
    args = parser.parse_args()

    if not args.aime_jsonl.exists():
        print(f"ERROR: {args.aime_jsonl} not found", file=sys.stderr)
        return 2

    aime_summary = aggregate(args.aime_jsonl, label="aime_2018_2023")
    synthetic_summary = (
        aggregate(args.synthetic_jsonl, label="synthetic")
        if args.synthetic_jsonl and args.synthetic_jsonl.exists()
        else None
    )

    # Final gate-a decision per proposal §1.7 — both sub-tests must pass
    # on the AIME set; synthetic must also pass for memorization control.
    aime_passes = (
        aime_summary["gate_a_per_problem_60pct"]
        and aime_summary["gate_a_means_thresholds"]
        and aime_summary["gate_a_sign_test_alpha_05"]
    )
    synthetic_passes = (
        None if synthetic_summary is None
        else (
            synthetic_summary["gate_a_per_problem_60pct"]
            and synthetic_summary["gate_a_means_thresholds"]
        )
    )

    decision = {
        "aime": aime_summary,
        "synthetic": synthetic_summary,
        "aime_gate_a_pass": aime_passes,
        "synthetic_gate_a_pass": synthetic_passes,
        "memorization_control_consistent": (
            synthetic_passes is None or synthetic_passes == aime_passes
        ),
        "final_gate_a_pass": aime_passes and (
            synthetic_passes is None or synthetic_passes
        ),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(decision, indent=2) + "\n")

    print(json.dumps(decision, indent=2))
    print()
    print(f"Decision written to {args.out}")
    print(f"FINAL gate (a) PASS: {decision['final_gate_a_pass']}")
    if not decision["final_gate_a_pass"]:
        print("Pivot per proposal §10 — discriminator–generator asymmetry rejected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

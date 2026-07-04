"""Week-1 premise pilot orchestrator (proposal §1.7).

This script *prints* the exact command sequence for the pilot —
including dollar estimates and approval prompts — but does NOT
launch any Modal jobs itself. GPU jobs require explicit per-launch
approval; this script makes the launch sequence transparent so the
operator can copy/paste each ``modal run`` line.

Run:

    python scripts/run_pilot.py

Phases:

1. **Local prep (no Modal spend).** Run ``fetch_data.py`` and
   ``generate_synthetic_problems.py`` to build the pilot corpora.

2. **Mining: pass@1024 sweep.** First Modal GPU job. Determines which
   of the 100 AIME 2018-2023 problems are in ``H_K_pilot`` (i.e.
   ``base_pass@1024 == 0``). Estimated cost ≈ $4–8.

3. **Premise test (gate a, AIME).** 4 conditions × 8 samples on the
   AIME pilot subset. Estimated cost ≈ $0.10.

4. **Premise test (gate a, synthetic memorization control).** Same
   protocol on the 50 synthetic problems. Estimated cost ≈ $0.05.

5. **Aggregate gate (a) decision.** Local-CPU script that combines the
   AIME + synthetic results into the pre-registered decision rule.

6. **Gate (b) — pass@8 crossover.** Requires a 1.5B-REFLEX-1000step
   checkpoint. The hybrid-latent GRPO trainer is in development;
   this script flags the dependency and stops here — the user
   completes phase 6 once the trainer lands.

The user reads the printed plan, decides, and copies one ``modal run``
command at a time.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _check(path: Path, what: str) -> bool:
    exists = path.exists()
    mark = "OK" if exists else "MISSING"
    print(f"  [{mark}] {what}: {path}")
    return exists


def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def main() -> int:
    banner("REFLEX-RLVR Week-1 premise pilot — pre-flight")
    print()

    print("Phase 0 (local prep):")
    aime_jsonl = REPO_ROOT / "data" / "aime" / "h_k_pilot.jsonl"
    synthetic_jsonl = REPO_ROOT / "data" / "aime" / "synthetic_pilot.jsonl"
    aime_full = REPO_ROOT / "data" / "aime" / "aime_1983_2024.jsonl"
    numina_jsonl = REPO_ROOT / "data" / "aime" / "numina_amc_aime.jsonl"

    have_aime = _check(aime_jsonl, "H_K_pilot (AIME 2018-2023 + Numina solutions)")
    have_synthetic = _check(synthetic_jsonl, "synthetic memorization-control set")
    _check(aime_full, "raw qq8933 AIME 1983-2024")
    _check(numina_jsonl, "raw NuminaMath amc_aime slice")

    if not have_aime:
        print()
        print("To build H_K_pilot:")
        print("  python scripts/fetch_data.py")
    if not have_synthetic:
        print()
        print("To build the synthetic control:")
        print("  python scripts/generate_synthetic_problems.py")

    if not (have_aime and have_synthetic):
        print()
        print("ABORT: pre-flight failed. Build the pilot corpora before launching Modal jobs.")
        return 2

    banner("Phase 1: Modal pre-flight (CPU only, ≈ $0.001)")
    print(
        "Confirms image build, secrets, volumes are wired."
    )
    print()
    print("  modal run scripts/modal_smoketest.py")
    print()

    banner("Phase 2: Hard-set mining (1st real GPU job, ≈ $4–8 on 4×H100)")
    print(
        "Sweeps base_pass@1024 on Qwen2.5-1.5B over the 100 AIME 2018-2023\n"
        "problems. Filters to H_K_pilot (problems with 0/1024 correct)."
    )
    print()
    print("  modal run scripts/run_mining.py \\")
    print(f"      --problems-jsonl data/aime/h_k_pilot.jsonl \\")
    print("      --model-id Qwen/Qwen2.5-1.5B \\")
    print("      --k 1024")
    print()
    print("Outputs (Modal cache volume):")
    print("  /cache/mining/h_k_pilot_pass1024.jsonl")
    print()

    banner("Phase 3: Premise test on AIME pilot (gate a, ≈ $0.10)")
    print(
        "4 conditions × 8 samples per problem on Qwen2.5-1.5B.\n"
        "Conditions: no_cot, oracle, corrupted, shuffled."
    )
    print()
    print("  modal run scripts/run_premise_test.py \\")
    print(f"      --problems-jsonl data/aime/h_k_pilot.jsonl \\")
    print("      --model-id Qwen/Qwen2.5-1.5B \\")
    print("      --n-samples 8 \\")
    print("      --save-to /cache/premise/premise_test_1p5b.jsonl")
    print()

    banner("Phase 4: Premise test on synthetic control (≈ $0.05)")
    print(
        "Same 4-condition protocol on the 50 synthetic problems.\n"
        "Memorization control — gate (a) must also pass on synthetic."
    )
    print()
    print("  modal run scripts/run_premise_test.py \\")
    print(f"      --problems-jsonl data/aime/synthetic_pilot.jsonl \\")
    print("      --model-id Qwen/Qwen2.5-1.5B \\")
    print("      --n-samples 8 \\")
    print("      --save-to /cache/premise/premise_test_synthetic.jsonl")
    print()

    banner("Phase 5: Aggregate gate (a) decision (local CPU)")
    print(
        "Pulls the per-problem JSONL files from the cache volume to local,\n"
        "runs the paired sign test α=0.05, decides gate (a)."
    )
    print()
    print("  # 5.1 Pull both result files from /cache/premise/ to local")
    print("  modal volume get reflex-rlvr-cache premise/ results/pilot/")
    print()
    print("  # 5.2 Decide")
    print("  python scripts/decide_gate_a.py \\")
    print("      --aime-jsonl results/pilot/premise/premise_test_1p5b.jsonl \\")
    print("      --synthetic-jsonl results/pilot/premise/premise_test_synthetic.jsonl")
    print()

    banner("Phase 6: pass@8 crossover gate (b) — IN-PROGRESS")
    print(
        "Status: requires the hybrid-latent GRPO trainer (1000 RL steps on\n"
        "Qwen2.5-1.5B, S_max=32, eps_max=0.10) which is the next\n"
        "engineering deliverable. This pilot covers gates (a) only.\n"
        "\n"
        "Once the trainer lands:\n"
        "  - 1000 RL steps on H_K_pilot, ≈ $80–120\n"
        "  - pass@8 measurement on the trained checkpoint vs base, ≈ $0.20\n"
        "  - Pivot rule per proposal §1.7:\n"
        "      ratio ≥ 1.5×: proceed with S_max=32 to full 1.5B run.\n"
        "      1.0× ≤ ratio < 1.5×: pivot to S_max=16, re-test 500 RL steps.\n"
        "      ratio < 1.0×: diagnostic ladder → halve eps_max → S_max=8 → ABORT 7B."
    )
    print()

    banner("Total estimated pilot cost: ≈ $5–10 for gates (a)")
    print(
        "Phases 1–5 (gates (a)) are the runnable pre-Modal envelope.\n"
        "Phase 6 (gate (b)) is gated on the GRPO trainer landing — see\n"
        "configs/pilot.yaml gate_b section for the full pre-registered protocol."
    )
    print()
    print("Every modal run command should be reviewed before execution.")
    print("Copy one line at a time, confirm cost, run.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())

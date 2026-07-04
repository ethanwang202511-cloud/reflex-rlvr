# Gate (a) decision — Week-1 premise pilot

Aggregator output of `scripts/decide_gate_a.py`. The headline numerical artifact of the premise pilot.

## Provenance

- **Source script.** `scripts/decide_gate_a.py` (local CPU; no Modal).
- **Inputs.** `results/pilot/premise/premise_test_1p5b.jsonl`, `results/pilot/premise/premise_test_synthetic.jsonl`.
- **Date.** 2026-05-03.
- **Cost.** $0 (local Python).

## Decision rule (proposal §1.7, locked)

Per-problem `passes_gate := p_disc_oracle ≥ 0.5 AND (p_disc_oracle - p_disc_corrupted) ≥ 0.2`. Gate (a) PASSES iff:
1. `≥ 60%` of problems pass per-problem, AND
2. mean(`p_disc_oracle`) ≥ 0.5 AND mean delta ≥ 0.2 (means thresholds), AND
3. paired sign test on per-problem deltas rejects H0:median=0 at α=0.05.

Both AIME and the synthetic memorization control must pass.

## Result

| metric                            | AIME H_K (51) | Synthetic (50) |
|-----------------------------------|---------------|----------------|
| n_passing_per_problem             | 0             | 1              |
| pct_passing_per_problem           | 0.0%          | 2.0%           |
| mean_p_disc_oracle                | 0.0123        | 0.160          |
| mean_p_disc_corrupted             | 0.0123        | 0.165          |
| delta_means                       | 0.0000        | -0.005         |
| sign-test p-value                 | 1.0           | 0.72           |
| ≥ 60% per-problem?                | False         | False          |
| means thresholds met?             | False         | False          |
| sign test rejects H0 at α=0.05?   | False         | False          |

`final_gate_a_pass`: **False**.
`memorization_control_consistent`: True (both fail consistently).

## Pivot per proposal §1.7 (pre-registered)

> "mean_p_disc(oracle) < 0.5 on AIME: rejects the discriminator–generator asymmetry premise. Note in overview/status.md that the project re-frames as a negative-result paper sharpening Yue et al."

Both failure modes of §1.7 trigger:
1. mean p_disc(oracle) = 0.0123 << 0.5 on AIME (and 0.16 << 0.5 on synthetic).
2. delta(oracle - corrupted) = 0.0 << 0.2 on AIME (and -0.005 << 0.2 on synthetic).

Per the locked pivot: **the project re-frames as a negative-result paper sharpening Yue et al.** Gate (b) (pass@8 crossover) is moot under this pivot — the latent register cannot rescue a teacher-free RL loop whose underlying premise (base discriminator > base generator) does not hold at the 1.5B scale.

## Methodological caveat (worth flagging in a follow-up, not a pivot blocker)

The exact equality `p_disc_oracle == p_disc_corrupted` across all 51 AIME problems is striking. The corruption deliberately preserves the final answer (`mining.py:73 candidates = nums[:-1]`), so a model that ignores intermediate steps and just copies the boxed answer would score identically on oracle and corrupted. However the absolute scores being near-zero (1.23%) refutes the "just copying" hypothesis — if the model were copying, p_disc would be near 1.0. The combined picture: the base ignores the provided solution AND fails to solve from scratch. This is a stronger negative result than a methodology bug. The proposal's own diagnostic (`p_disc(shuffled) ≈ p_disc(oracle)` ⇒ matching terminal cues) does not fire — shuffled (0.132) ≠ oracle (0.012) — which independently rules out the "terminal-cue matching" failure mode.

A follow-up test using LOG-PROBABILITY-based p_disc (rather than behavioral generation) would be more powerful — it would directly measure "does the model assign higher probability to the true answer given oracle vs corrupted?" — but that test was explicitly NOT pre-registered, so this gate (a) result stands as the formal pre-committed answer.

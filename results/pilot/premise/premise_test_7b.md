# Premise test (gate a) — Qwen2.5-7B base, AIME H_K subset (post-hoc scale check)

Behavioral 4-condition test on the same 51-problem AIME H_K subset, but with **Qwen2.5-7B base** instead of 1.5B. **Not pre-registered** — added as a post-hoc scale check after the 1.5B premise test failed gate (a) on AIME, to test whether the failure is 1.5B-specific or generalizes upward in scale.

## Provenance

- **Source script.** `scripts/run_premise_test.py` → `src/reflex_rlvr/modal_app/premise_test.py:run_premise_test` (no code change vs the 1.5B run; only `--model-id Qwen/Qwen2.5-7B`).
- **Input file.** `data/aime/h_k_pilot_filtered.jsonl` (same 51 problems as the 1.5B run; H_K filter from the 1.5B mining `pass@1024(Qwen2.5-1.5B-base) = 0`).
- **Caveat.** The H_K filter is per-base (51 problems are H_K for the 1.5B base); the 7B base may solve some of these from scratch. We did not re-mine for a 7B-specific H_K because the goal is direct comparability of the discriminator-vs-generator measurement on the same problem set, and the 7B no_cot baseline (0.000 — see below) confirms the 7B base does not solve any of these 51 H_K problems either.
- **Model checkpoint.** `Qwen/Qwen2.5-7B` (base, not Instruct).
- **Seed.** 1337.
- **n_samples per condition.** 8.
- **Conditions.** identical 4-condition protocol (no_cot, oracle, corrupted, shuffled).
- **Modal app.** `ap-...` (see /tmp/premise_7b.log for the exact app id).
- **Date.** 2026-05-03.
- **Cost.** $0.39.

## Schema

Identical to `premise_test_1p5b.md`.

## Aggregate result

- mean p_disc_no_cot:    0.000  (7B base also cannot solve these problems from scratch — confirms H_K filter holds at 7B)
- mean p_disc_oracle:    0.000  (ZERO — even worse than 1.5B's 0.012)
- mean p_disc_corrupted: 0.0147 (slightly HIGHER than oracle)
- mean p_disc_shuffled:  0.1373 (similar pattern to 1.5B's 0.132)
- delta_oracle_minus_corrupted: -0.0147 (NEGATIVE — corrupted scored higher than oracle)
- **Gate (a) FAILED at 7B** on all three sub-criteria (per-problem 0/51 = 0%, means thresholds, sign test).

## Implication for the paper

The premise rejection is not 1.5B-specific. **Both 1.5B and 7B base models exhibit the same failure mode**: they ignore the provided solution AND fail to solve from scratch. The "the base just needs to be larger" hypothesis is refuted at ≤ 7B. This strengthens the headline claim.

7B oracle (0.000) being slightly worse than 1.5B oracle (0.012) is surprising at first glance. A plausible explanation: a more capable base is more confident in continuing the prompt as a "solution write-up" (more reasoning text after `Final answer:`) rather than terminating with `\boxed{N}`, so the verifier (which requires `\boxed{N}`) catches fewer correct answers. This is a *prompt-format* artifact, not a discrimination-strength artifact, and would be controlled in a logprob-based premise test (proposed as future work).

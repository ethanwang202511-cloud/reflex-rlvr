# Premise test (gate a) — synthetic memorization control

Behavioral 4-condition test on the 50 fresh synthetic problems (parameterized algebra/combinatorics, SymPy round-trip-verified). Per proposal §1.7 this is the memorization-control test that runs alongside AIME H_K.

## Provenance

- **Source script.** `scripts/run_premise_test.py` → `src/reflex_rlvr/modal_app/premise_test.py:run_premise_test`.
- **Input file.** `data/aime/synthetic_pilot.jsonl` (50 GPT-4o-style synthetic AIME-format problems).
- **Model checkpoint.** `Qwen/Qwen2.5-1.5B` (base).
- **Seed.** 1337.
- **n_samples per condition.** 8.
- **Conditions.** identical 4-condition protocol as the AIME premise test.
- **Modal app.** `ap-Fyyq...` (synthetic re-run; first attempt `ap-opYQjxKrpiL3YIFIsrEPxG` errored on a transient HF ConnectionResetError before reaching CUDA graph capture, $0 cost).
- **Date.** 2026-05-03.
- **Cost.** $0.25.

## Schema

Identical to `premise_test_1p5b.md` schema.

## Aggregate result

- mean p_disc_no_cot:    0.135
- mean p_disc_oracle:    0.160
- mean p_disc_corrupted: 0.165 (slightly HIGHER than oracle)
- mean p_disc_shuffled:  0.070 (lower than no_cot — shuffling actively hurts on synthetic)
- delta_oracle_minus_corrupted: -0.005 (NEGATIVE)
- **Per-problem pass rate: 1/50 = 2%** — well below 60%.
- **Paired sign test:** n_pos=12, n_neg=14, p=0.72, fail to reject H0.
- **Gate (a) synthetic: FAILED** on all three sub-criteria.
- **Memorization control consistent:** both AIME and synthetic fail → the AIME failure is not a memorization confound.

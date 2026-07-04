# Premise test (gate a) — AIME H_K subset

Behavioral 4-condition test of the discriminator–generator asymmetry premise (proposal §1.7) on the 51-problem `H_K_pilot ⊂ AIME 2018–2023` (problems with `pass@1024(base) = 0` per the mining run).

## Provenance

- **Source script.** `scripts/run_premise_test.py` → `src/reflex_rlvr/modal_app/premise_test.py:run_premise_test`.
- **Input file.** `data/aime/h_k_pilot_filtered.jsonl` (51 problems = the H_K subset of `data/aime/h_k_pilot.jsonl`, filtered locally on `mining/h_k_pilot_pass1024.jsonl[n_correct] == 0`).
- **Model checkpoint.** `Qwen/Qwen2.5-1.5B` (base).
- **Seed.** 1337.
- **n_samples per condition.** 8.
- **Conditions.** `no_cot` (problem only), `oracle` (problem + ground-truth solution), `corrupted` (one intermediate numeric step changed, final answer preserved), `shuffled` (solution paragraphs randomly reordered).
- **Modal app.** `ap-FzHNDHlG2mqEGAhk6kKHQe`.
- **Date.** 2026-05-03.
- **Cost.** $0.24.

## Schema

| field | type | meaning |
|---|---|---|
| id | str | problem id (matches mining/h_k_pilot_pass1024.jsonl) |
| year | int | competition year |
| answer | int | ground-truth integer |
| p_disc_no_cot | float | fraction of 8 samples correct, problem-only prompt |
| p_disc_oracle | float | fraction correct, problem + oracle solution |
| p_disc_corrupted | float | fraction correct, problem + corrupted solution |
| p_disc_shuffled | float | fraction correct, problem + shuffled solution |

## Aggregate result

- mean p_disc_no_cot:    0.0000
- mean p_disc_oracle:    0.0123
- mean p_disc_corrupted: 0.0123 (identical to oracle to 4 decimal places)
- mean p_disc_shuffled:  0.1324
- delta_oracle_minus_corrupted: 0.0000
- **Per-problem pass rate (≥0.5 oracle AND ≥0.2 delta): 0/51 = 0%** — well below the 60% pre-registered threshold.
- **Paired sign test:** n_eff=0 (no problems had nonzero delta), p=1.0, fail to reject H0.
- **Gate (a) AIME: FAILED** on all three sub-criteria.

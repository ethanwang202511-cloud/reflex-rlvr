# Hard-set mining — h_k_pilot_pass1024.jsonl

Per-problem pass@1024 sweep on Qwen2.5-1.5B base over the 69 AIME 2018–2023 pilot problems.

## Provenance

- **Source script.** `scripts/run_mining.py` (Modal entrypoint forwarding to `src/reflex_rlvr/modal_app/mining.py:mine_pass_at_k`).
- **Model checkpoint.** `Qwen/Qwen2.5-1.5B` (base, not Instruct).
- **Seed.** 1337 (incremented per chunk: 1337, 1338, ..., 1400 for the 64 chunks).
- **Modal app.** `ap-rRv37WPJKQM2jqYSd2U3Lu` (mining k=1024).
- **Modal job ID.** `mine_pass_at_k` function in `reflex-rlvr-mining` app.
- **Date.** 2026-05-03.
- **Cost.** $2.67 (gen_elapsed=925.52s on 4×H100, total=1006.39s).
- **Patch context.** Generation chunked into 16-sample batches (mining.py:91-141 patched 2026-05-03 after a 31-min vLLM scheduler stall on the original n=1024 single-call pattern; chunked patch validated end-to-end via the k=128 dry-run at $0.54 before the k=1024 launch).

## Schema

Each row = one problem.

| field | type | meaning |
|---|---|---|
| id | str | AIME problem id, e.g. `2018-II-1` |
| year | int | competition year |
| problem_number | int | problem index within the competition |
| answer | int | ground-truth integer answer (0-999 per AIME format) |
| n_samples | int | k = 1024 |
| n_correct | int | number of the 1024 samples whose `\boxed{N}` matched the ground truth |
| samples_correct | list[int] | per-sample 0/1 correctness flags (length 1024) |

## Result

- 69 problems processed.
- **n_solved_any: 18** (problems where the base solved at least once at pass@1024).
- **n_solved_zero: 51** → these define **`H_K_pilot`** (the pre-registered hard set per proposal §1.7).
- The non-H_K 18 had n_correct distribution `[1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 8, 10]` — most are borderline (1/1024).

## Companion file

`h_k_pilot_pass128_dryrun.jsonl` — same format, k=128 dry-run that validated the chunked-generation patch ($0.54, n_solved_any=3, n_solved_zero=66).

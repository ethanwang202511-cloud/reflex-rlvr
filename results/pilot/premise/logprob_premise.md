# Logprob-based premise test (multi-base + R1-distill follow-up to behavioral gate (a))

A more sensitive logprob-based version of the §1.7 behavioral premise test, run on the same 51 AIME H_K problems. **Not pre-registered**; added in waves to test whether the behavioral negative result holds under measurement, scale, family, post-training, and reasoning-distillation extensions.

## Provenance

- **Source script.** `scripts/run_logprob_premise.py` → `src/reflex_rlvr/modal_app/premise_test.py:run_logprob_premise_test`.
- **Input file.** `data/aime/h_k_pilot_filtered.jsonl` (51 AIME H_K problems).
- **Seed.** 1337 (deterministic — temperature=0.0 for logprob scoring).
- **Conditions.** identical 4-condition protocol as the behavioral test.
- **v0.4 wave (2026-05-03, $0.48):** Qwen2.5-1.5B base, Qwen2.5-7B base. Modal apps `ap-ykwIGFKDo35oqMw683Rebv`, `ap-j1uEYXULVPi4Q0MAWNnkWg`.
- **v0.4-D6 wave (2026-05-04, $0.40):** Qwen2.5-{1.5B,7B}-Instruct (post-training contrast).
- **v0.5 wave (2026-05-04, $0.30):** Llama-3.1-8B base (cross-family).
- **v0.6 wave (2026-05-04, ~$3-5):** Qwen2.5-72B base (scale-out), Mistral-7B-v0.3 base, Gemma-2-9B base (cross-family broadening), DeepSeek-R1-Distill-Qwen-{1.5B,7B}, DeepSeek-R1-Distill-Llama-8B (reasoning-distillation triple). Modal apps `ap-Dy3DiXYUHplrUVzsPuNZ8Z` (R1-Qwen-7B), `ap-Tvp5SHrikIjPNOwU5vVFWm` (R1-Qwen-1.5B), `ap-PvhKW1CXzo6uKzCcJPmibK` (R1-Llama-8B), `ap-YIbgu0K9PrJxH3qYQqWSMm` (Mistral-7B), `ap-CEqObcVVv8ttDfS0PBpZVQ` (Gemma-2-9B), `ap-nkcHt82xOJNerkrzf01OD5` (Qwen-72B).

## Method

For each (problem, condition), construct prompt + answer_string where answer_string = `\boxed{N}` (the same format the verifier requires). Compute log P(answer_string | prompt) via vLLM's `prompt_logprobs=0, max_tokens=1, temperature=0.0`. Aggregate per-token-mean logprob per problem per condition. Sign test on per-problem deltas (oracle − corrupted) via `scipy.stats.wilcoxon(..., alternative="greater")`.

Decision rule (logprob analogue of behavioral gate (a)):
- mean delta(oracle − corrupted) > 0
- > 60% of problems have logp(oracle) > logp(corrupted)
- Wilcoxon one-sided p < 0.05

## Schema (per-problem JSONL)

| field | type | meaning |
|---|---|---|
| id | str | problem id |
| year | int | competition year |
| answer | int | ground-truth answer |
| answer_string | str | `\boxed{N}` |
| n_answer_tokens | int | tokens in `\boxed{N}` (typically 4–7 after BPE) |
| logp_no_cot | float | mean per-token logprob of answer given problem only |
| logp_oracle | float | mean per-token logprob of answer given oracle solution |
| logp_corrupted | float | mean per-token logprob given corrupted solution |
| logp_shuffled | float | mean per-token logprob given shuffled solution |
| delta_oracle_corrupted | float | logp_oracle − logp_corrupted |
| oracle_beats_corrupted | bool | delta > 0 |

## Aggregate result

| model                              | mean logp(oracle) | mean logp(corrupted) | Δ(oracle−corr.) | pct(oracle beats) | Wilcoxon p | gate |
|------------------------------------|-------------------|----------------------|-----------------|-------------------|------------|------|
| Qwen2.5-1.5B base                  | -0.5832           | -0.5808              | -0.0024         | 56.86%            | 0.554      | FAIL |
| Qwen2.5-7B base                    | -0.8252           | -0.8181              | -0.0071         | 47.06%            | 0.928      | FAIL |
| Qwen2.5-1.5B-Instruct              | -0.7019           | -0.7003              | -0.0016         | 50.98%            | n/a        | FAIL |
| Qwen2.5-7B-Instruct                | -0.9612           | -0.9533              | -0.0079         | 50.98%            | n/a        | FAIL |
| Llama-3.1-8B base                  | -1.0022           | -0.9948              | -0.0074         | n/a               | 0.97       | FAIL |
| **Qwen2.5-72B base** (v0.6)        | -0.6937           | -0.6742              | -0.0194         | 37.25%            | 0.993      | FAIL |
| **Mistral-7B-v0.3 base** (v0.6)    | -0.4761           | -0.4734              | -0.0027         | 54.9%             | 0.265      | FAIL |
| **Gemma-2-9B base** (v0.6)         | -0.5813           | -0.5767              | -0.0046         | 56.86%            | 0.584      | FAIL |
| **R1-Distill-Qwen-1.5B** (v0.6)    | -1.2492           | -1.2372              | -0.0120         | 27.45%            | 1.000      | FAIL |
| **R1-Distill-Qwen-7B** (v0.6)      | -1.3953           | -1.3906              | -0.0047         | 47.06%            | 0.784      | FAIL |
| **R1-Distill-Llama-8B** (v0.6)     | -2.2306           | -2.2242              | -0.0064         | 49.02%            | 0.688      | FAIL |
| **gate criterion**                 | —                 | —                    | **> 0**         | **> 60%**         | **< 0.05** | —    |

**Every confirmed measurement FAILS the logprob gate.** R1-Distill-Qwen-1.5B is the strongest reverse signal (corrupted beats oracle on 72.5% of problems; Cohen's d = -0.56 paired).

## Implications

1. **Confirmation under a more sensitive measurement.** The behavioral gate (a) test could in principle be confounded by prompt-format effects (model continues solution prose rather than terminating with `\boxed{N}`); the logprob test does not have this confound. It still rejects the discrimination premise. The negative result is robust to the methodology choice.
2. **The base IS conditioning on the prefix.** logp_no_cot is much worse than the chain-conditioned values (-2.10 vs -0.58 at 1.5B; -1.68 vs -0.82 at 7B). The base model's confidence in `\boxed{N}` increases substantially when ANY long context is provided. It just doesn't increase *more* when that context is correct vs corrupted.
3. **At 7B base, the asymmetry actually goes slightly negative.** corrupted beats oracle on 52.94% of problems at 7B (p_disc(corrupted)=−0.818 > p_disc(oracle)=−0.825). The Wilcoxon one-sided p of 0.928 means we should completely accept H0 (no asymmetry) and in fact mildly favor H1 in the *opposite* direction.
4. **shuffled also helps at both scales.** logp_shuffled is the highest of the four conditions at 7B (−0.7629). This is consistent with the behavioral test's shuffled > oracle pattern. The base treats the chain content as ~independent of correctness; what matters is having SOME long math-shaped context.

This sharpens the headline negative result: at 1.5B and 7B base, on hard math, there is **no measurable behavioral or logprob discrimination of correct vs. corrupted chains**. The premise that motivates teacher-free RLVR via the discriminator route is empirically rejected.

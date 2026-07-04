# v0.7 measurements provenance (2026-05-07, iter-D9)

This document records the four new measurement types added in v0.7,
addressing NeurIPS-calibrated reviewer feedback on v0.6.

## Reviewer concern addressed

The v0.6 logprob test corrupted only an intermediate numeric token but
preserved the final $\backslash$boxed{N}. Both oracle and corrupted
prompts therefore displayed the *same* visible final answer, and the
test scored logp of that one answer string under both conditions. This
made chain-content sensitivity invisible by construction — any logp Δ
≈ 0 finding is "of course Δ ≈ 0" regardless of whether the model
actually parses the chain.

## v0.7 measurement types

### 1. Answer-relabeled (consistent) corruption logprob — `consistent_corruption_*.jsonl`

- Code: `src/reflex_rlvr/modal_app/discrimination_v2.py::run_consistent_corruption_logprob`
- Driver: `scripts/run_consistent_corruption.py`
- Protocol: corrupt one intermediate numeric token AND replace the
  final $\backslash$boxed{N} with $\backslash$boxed{N+δ} using the
  same shift δ. Score logp of two answer strings under each condition:
  $\backslash$boxed{N_oracle} and $\backslash$boxed{N_corrupted}.
  Compute $\sigma_{\rm disc} = \delta_{\rm disc}^{\rm oracle} -
  \delta_{\rm disc}^{\rm corrupted}$.
- Models: Qwen2.5-1.5B base, Qwen2.5-7B base, Qwen2.5-7B-Instruct,
  DeepSeek-R1-Distill-Qwen-1.5B, Qwen2.5-72B base.
- Cost: \$0.27 (R1-1.5B) + \$0.29 (Qwen-1.5B) + \$0.39 (7B) + \$0.38
  (7B-Instruct) + \$2.41 (72B) = \$3.74.
- Modal apps: ap-XJbmg7De4XLxAtsmFFrXyJ (1.5B), ...

| model | $\delta_o$ | $\delta_c$ | $\sigma_{\rm disc}$ | pct $\delta_o > \delta_c$ |
|---|---:|---:|---:|---:|
| Qwen2.5-1.5B base | $+1.39$ | $-1.19$ | $\mathbf{2.58}$ | $\mathbf{100\%}$ |
| Qwen2.5-7B base | $+1.59$ | $-0.91$ | $\mathbf{2.50}$ | $\mathbf{100\%}$ |
| Qwen2.5-7B-Instruct | $+2.09$ | $-1.20$ | $\mathbf{3.29}$ | $\mathbf{100\%}$ |
| R1-Distill-Qwen-1.5B | $+1.14$ | $-0.91$ | $\mathbf{2.05}$ | $\mathbf{100\%}$ |
| Qwen2.5-72B base | $+1.66$ | $-0.72$ | $\mathbf{2.38}$ | $\mathbf{100\%}$ |

Wilcoxon one-sided $p < 10^{-9}$ on every measurement. **Gate passes
5/5.** The base IS sensitive to chain content; v0.6 logp test held
visible answer constant and could not see this.

### 2. Verdict (YES/NO) discrimination test — `verdict_*.jsonl`

- Code: `src/reflex_rlvr/modal_app/discrimination_v2.py::run_verdict_test`
- Driver: `scripts/run_verdict_test.py`
- Protocol: prompt the model with a math-grader template ("Is the
  proposed solution correct? Answer YES or NO. / Answer:"), score
  logp(" YES") and logp(" NO"), compute $\ell = $ logp(YES) − logp(NO)
  and $\Delta_{\rm verdict} = \ell^{\rm oracle} - \ell^{\rm corrupted}$.
- Conditions per problem: oracle, consistent_corrupted, shuffled.
- Models: same 5 as above.
- Cost: \$0.24 (1.5B) + \$0.39 (7B) + \$0.40 (7B-Instruct) + \$0.25
  (R1-1.5B) + \$1.98 (72B) = \$3.26.

| model | $\bar\ell^{\rm oracle}$ | $\bar\ell^{\rm corrupted}$ | $\bar\ell^{\rm shuffled}$ | $\Delta_{\rm verdict}$ | pct $o>c$ |
|---|---:|---:|---:|---:|---:|
| Qwen2.5-1.5B base | $-0.12$ | $-0.66$ | $-0.69$ | $\mathbf{0.53}$ | $\mathbf{76.5\%}$ |
| Qwen2.5-7B base | $+0.68$ | $-0.70$ | $-0.01$ | $\mathbf{1.39}$ | $\mathbf{80.4\%}$ |
| Qwen2.5-7B-Instruct | $+0.08$ | $\mathbf{-7.26}$ | $-2.89$ | $\mathbf{7.34}$ | $\mathbf{92.2\%}$ |
| R1-Distill-Qwen-1.5B | $-1.24$ | $-1.39$ | $-1.24$ | $\mathbf{0.15}$ | $66.7\%$ |
| Qwen2.5-72B base | $+2.73$ | $+1.18$ | $+1.65$ | $\mathbf{1.55}$ | $\mathbf{92.2\%}$ |

Wilcoxon one-sided $p \leq 1.5 \times 10^{-5}$ on every measurement.
**Gate passes 5/5.** Instruct-tuning multiplies $\Delta_{\rm verdict}$
5× (1.39 → 7.34 at 7B).

### 3. Prompt-format ablation — `format_{chat,interleaved}_qwen_1p5b.jsonl`

- Code: `src/reflex_rlvr/modal_app/discrimination_v2.py::run_format_ablation_logprob`
- Driver: `scripts/run_format_ablation.py`
- Protocol: re-run the original v0.6 logprob protocol (`corrupt_solution_one_step`,
  score $\backslash$boxed{N_oracle}) under three prompt templates:
  bare (v0.6), chat (`tokenizer.apply_chat_template`), interleaved
  ($<$problem$><$/problem$><$solution_sketch$>...<$/solution_sketch$><$answer$>$).
- Model: Qwen2.5-1.5B base.
- Cost: \$0.26 (chat) + \$0.26 (interleaved) = \$0.52.

| template | $\bar p_{\rm oracle}$ | $\bar p_{\rm corrupted}$ | $\Delta_{\rm logp}$ | pct $o > c$ |
|---|---:|---:|---:|---:|
| bare (v0.6) | $-0.583$ | $-0.581$ | $-0.0024$ | $56.9\%$ |
| chat | $-1.981$ | $-1.980$ | $-0.0004$ | $41.2\%$ |
| interleaved | $-1.497$ | $-1.491$ | $-0.0062$ | $47.1\%$ |

All three templates produce $\Delta \approx 0$ — v0.6 result is robust
to template, ruling out the format-effect hypothesis. The shuffled-
higher pattern persists under all three templates, supporting the
surface-feature attention reading.

### 4. Per-token surprise mechanistic probe — `per_token_surprise_qwen_1p5b.jsonl`

- Code: `src/reflex_rlvr/modal_app/discrimination_v2.py::run_per_token_surprise`
- Driver: `scripts/run_per_token_surprise.py`
- Protocol: for each problem, build oracle and consistent_corrupted
  prompts; run vLLM `prompt_logprobs=0, max_tokens=1`; extract per-
  token surprise (= -logp); identify token positions where the two
  prompts differ (intermediate corruption + final boxed corruption);
  build $\pm 3$-token windows around diff sites; compute mean surprise
  inside vs outside window. LMG = (corrupted_window − oracle_window) −
  (corrupted_elsewhere − oracle_elsewhere).
- Model: Qwen2.5-1.5B base.
- Cost: \$0.25.

| metric | value |
|---|---:|
| mean LMG (logp units) | $\mathbf{+1.26}$ |
| pct positive (51 problems) | $\mathbf{100\%}$ (51/51) |
| Wilcoxon one-sided $p$ | $< 10^{-9}$ |
| n_skipped | $0$ |

The model IS locally surprised at corruption sites by 1.26 logp units
more than at non-corruption positions. Mechanistic confirmation of the
v0.6 → v0.7 retraction: chain-content sensitivity exists at the per-
token level and propagates through any pathway that does not anchor
on a visibly-displayed answer.

## Code review record

Pre-launch code review caught:
- **HIGH** — NaN-contaminated JSONL would produce invalid JSON. Fixed:
  store NaN as None at write time (`_sanitize_row`) + `allow_nan=False`
  at write site + NaN filter in `_mean`.
- **HIGH** — `corrupt_solution_consistent` had a latent self-
  consistency bug for small answers: the intermediate corruption used
  `delta` but the final answer was bumped via a different path. Fixed
  by adjusting `delta` itself (not just the result) before applying
  to both intermediate and final.
- **MEDIUM** — `corrupt_solution_consistent` first-match instead of
  last-match for `\boxed{}` replacement when multiple matched. Fixed
  by `last match wins` loop.

Pre-launch second review of `run_per_token_surprise` caught:
- **BLOCKER** — oracle window construction clamped every corrupted-
  window index to oracle bounds, producing systematic bias whenever
  oracle and corrupted lengths differed. Fixed by building oracle
  window independently around the same diff centers, clamped to
  oracle bounds first.
- **HIGH** — missing fallback when `pos_lp.get(actual_id) is None`.
  Fixed by mirroring reference's `next(iter(pos_lp.values()))` fallback.
- **MEDIUM** — tail-alignment for `last_diff` only marked `min_len`,
  missing the boxed-corruption site when in the longer sequence's
  tail. Fixed by `last_diff = max(last_diff, min_len)` when lengths
  differ.

## v0.7 wave total

\$10.49 across 13 Modal runs. ~3.4 H100·hr. Cumulative spend:
\$30.51 of \$500 user-approved ceiling.

"""Discrimination-v2 test suite (Week-1 pilot gate follow-ups).

Three Modal functions extending the logprob premise test:

A) ``run_consistent_corruption_logprob``
   Self-consistent corruption: both an intermediate numeric step AND the
   final ``\\boxed{N}`` are shifted by the same delta. Measures whether
   the model's log-prob discriminates oracle vs. consistently-wrong CoT.

B) ``run_verdict_test``
   Instruction-following grader: feed the model a YES/NO verdict prompt
   for three conditions (oracle, consistent_corrupted, shuffled) and
   compare logit(" YES") - logit(" NO") across conditions.

C) ``run_format_ablation_logprob``
   Re-runs the original logprob premise test under three prompt templates
   (bare, chat, interleaved) to check sensitivity to framing.

All three mirror the patterns from ``premise_test.py`` exactly:
- same imports, same app, same @app.function decorators
- same ``prompt_logprobs=0, max_tokens=1, temperature=0.0`` scoring trick
- same ``n_ans = len(full_ids) - len(prompt_ids)`` boundary
- same ``_mean`` aggregator
- same return-dict format with model_id, n_problems, elapsed, cost, save_to
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path


def _nan_to_none(v):
    """Convert nan/None floats to None for JSON-strict serialization."""
    if v is None:
        return None
    try:
        if isinstance(v, float) and math.isnan(v):
            return None
    except (TypeError, ValueError):
        pass
    return v


def _sanitize_row(row: dict) -> dict:
    """Replace NaN values in a per-problem row with None (RFC 8259-strict)."""
    return {k: _nan_to_none(v) for k, v in row.items()}

from reflex_rlvr.modal_app.common import (
    APP_NAME,
    IMAGE,
    VOLUMES,
    estimate_cost,
    get_secrets,
    gpu_spec,
)

try:
    import modal
except ImportError:  # noqa: BLE001
    modal = None  # type: ignore[assignment]


_LATEX_NUM = re.compile(r"\b(\d+)\b")
_BOXED = re.compile(r"\\boxed\{([^}]*)\}")


# ---------------------------------------------------------------------------
# A) Helper: corrupt_solution_consistent
# ---------------------------------------------------------------------------

def corrupt_solution_consistent(
    solution: str,
    answer,
    *,
    seed: int = 0,
) -> tuple[str, int]:
    """Corrupt a solution so that BOTH an intermediate step AND the final
    ``\\boxed{N}`` are shifted by the same delta. The chain reads as
    "self-consistent but wrong by delta".

    Parameters
    ----------
    solution:
        The oracle solution string.
    answer:
        The ground-truth integer answer (used to guarantee
        ``n_corrupted != original_answer``).
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    (new_solution_string, n_corrupted)
        ``n_corrupted`` is the integer that appears in the final
        ``\\boxed{...}`` after corruption.
    """
    import random

    rng = random.Random(seed)

    original_answer = int(answer)
    delta_choices = [1, 2, 3, 5, 7, -1, -2, -3, -5, -7]

    # Pick delta that avoids producing the same answer or a non-positive value.
    # Adjust delta itself (not just the result) so the SAME delta is used for
    # both the intermediate corruption and the final-answer relabel.
    delta = rng.choice(delta_choices)
    if original_answer + delta <= 0:
        delta = abs(delta)
    if original_answer + delta == original_answer:
        delta = delta + 1 if delta >= 0 else delta - 1
    n_corrupted = original_answer + delta
    # Hard guarantee.
    assert n_corrupted != original_answer and n_corrupted > 0

    # --- Step 1: corrupt one intermediate numeric token ---
    nums = list(_LATEX_NUM.finditer(solution))
    if nums:
        # Avoid corrupting the very last number (likely the terminal answer).
        candidates = nums[:-1] if len(nums) > 1 else nums
        target = rng.choice(candidates)
        original_num = int(target.group(1))
        new_num = max(0, original_num + delta)
        if new_num == original_num:
            new_num += 1
        span = target.span(1)
        solution = solution[: span[0]] + str(new_num) + solution[span[1] :]

    # --- Step 2: replace the final \\boxed{...} with \\boxed{n_corrupted} ---
    boxed_matches = list(_BOXED.finditer(solution))
    if boxed_matches:
        # Find LAST \boxed{} matching the original answer; else fall back to
        # the very last \boxed{} in the solution.
        target_match = None
        for m in boxed_matches:
            try:
                if int(m.group(1)) == original_answer:
                    target_match = m  # last match wins (do not break)
            except (ValueError, TypeError):
                pass
        if target_match is None:
            target_match = boxed_matches[-1]
        start, end = target_match.span()
        solution = (
            solution[:start]
            + f"\\boxed{{{n_corrupted}}}"
            + solution[end:]
        )
    else:
        # No \\boxed{} at all — append a final answer line.
        solution = solution + f"\nFinal answer: \\boxed{{{n_corrupted}}}\n"

    return solution, n_corrupted


# ---------------------------------------------------------------------------
# Re-export helpers from premise_test for use in the Modal functions below.
# Importing at module level is fine since premise_test has no heavy deps.
# ---------------------------------------------------------------------------

from reflex_rlvr.modal_app.premise_test import (  # noqa: E402
    corrupt_solution_one_step,
    shuffle_solution_steps,
)


# ---------------------------------------------------------------------------
# A2) Helper: corrupt_solution_random_answer (Round-7 reviewer-defense probe)
# ---------------------------------------------------------------------------

def corrupt_solution_random_answer(
    solution: str,
    answer,
    answer_pool: list,
    *,
    seed: int = 0,
) -> tuple[str, int]:
    """Replace the final ``\\boxed{N_oracle}`` with ``\\boxed{N_random}`` where
    ``N_random`` is a different in-distribution AIME-style integer drawn from
    ``answer_pool`` (other problems' ground-truth answers).

    Unlike ``corrupt_solution_consistent``, the chain content is left unchanged
    -- only the terminal boxed value is swapped. This isolates the question
    "does the model penalize a final answer that disagrees with what the
    intermediate chain steps actually compute?" from any out-of-distribution
    token-frequency artifact a delta-shifted answer might create.

    Parameters
    ----------
    solution:
        The oracle solution string.
    answer:
        The ground-truth integer answer for this problem.
    answer_pool:
        A list of plausible in-distribution AIME answers to draw the random
        substitute from. Typically the set of ground-truth answers from the
        other problems in the eval set.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    (new_solution_string, n_random)
        ``n_random`` is the integer that now appears in the final
        ``\\boxed{...}``.
    """
    import random

    rng = random.Random(seed)
    original_answer = int(answer)
    # Filter: exclude the original answer itself.
    choices = [int(a) for a in answer_pool if int(a) != original_answer]
    if not choices:
        # Fall back to original_answer + 1 if the pool is degenerate.
        n_random = original_answer + 1
    else:
        n_random = rng.choice(choices)
    assert n_random != original_answer

    boxed_matches = list(_BOXED.finditer(solution))
    if boxed_matches:
        target_match = None
        for m in boxed_matches:
            try:
                if int(m.group(1)) == original_answer:
                    target_match = m  # last match wins
            except (ValueError, TypeError):
                pass
        if target_match is None:
            target_match = boxed_matches[-1]
        start, end = target_match.span()
        solution = (
            solution[:start]
            + f"\\boxed{{{n_random}}}"
            + solution[end:]
        )
    else:
        solution = solution + f"\nFinal answer: \\boxed{{{n_random}}}\n"

    return solution, n_random


# ---------------------------------------------------------------------------
# Modal app registration
# ---------------------------------------------------------------------------

if modal is not None:
    app = modal.App(f"{APP_NAME}-discrimination-v2", image=IMAGE)

    # -----------------------------------------------------------------------
    # B) run_consistent_corruption_logprob
    # -----------------------------------------------------------------------

    @app.function(
        gpu=gpu_spec("pilot"),
        volumes=VOLUMES,
        secrets=get_secrets(),
        timeout=3600,
        memory=32_000,
    )
    def run_consistent_corruption_logprob(
        problems: list[dict],
        *,
        model_id: str = "Qwen/Qwen2.5-1.5B",
        seed: int = 1337,
        save_to: str = "/cache/premise/consistent_corruption_logprob.jsonl",
    ) -> dict:
        """Measure discrimination between oracle and consistently-wrong CoT.

        Three conditions per problem:
        - ``oracle``:              x ⊕ y_oracle
        - ``consistent_corrupted``: x ⊕ y_consistent (intermediate + final
          \\boxed both shifted by same delta)
        - ``no_cot``:              x alone

        For each (problem, condition) we score TWO answer strings:
        - ``\\boxed{N_oracle}``    → logp_correct
        - ``\\boxed{N_corrupted}`` → logp_wrong

        and compute delta_disc = logp_correct - logp_wrong.

        Gate: mean_discrimination_signal > 0
              AND pct_oracle_above_corrupted > 0.6
              AND Wilcoxon one-sided p < 0.05
        """
        from scipy.stats import wilcoxon  # noqa: PLC0415
        from vllm import LLM, SamplingParams  # noqa: PLC0415

        out_path = Path(save_to)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        llm = LLM(
            model=model_id,
            dtype="bfloat16",
            seed=seed,
            tensor_parallel_size=4,
            max_model_len=2048,
            gpu_memory_utilization=0.85,
            trust_remote_code=True,
        )
        tokenizer = llm.get_tokenizer()

        scoring_params = SamplingParams(
            prompt_logprobs=0,
            max_tokens=1,
            temperature=0.0,
            seed=seed,
        )

        def build_prompt(problem: str, cot: str | None) -> str:
            if cot is None or cot == "":
                return (
                    "Solve the following problem. Show your reasoning briefly, "
                    "then put the final integer answer in \\boxed{}.\n\n"
                    f"Problem: {problem}\n\nSolution: "
                )
            return (
                "Solve the following problem. Use the provided solution sketch "
                "as guidance. Put the final integer answer in \\boxed{}.\n\n"
                f"Problem: {problem}\n\nSolution sketch: {cot}\n\n"
                "Final answer: "
            )

        def score_answer(prompt: str, answer_string: str) -> float:
            """Return mean per-token log-prob of answer_string given prompt."""
            full_text = prompt + answer_string
            result = llm.generate([full_text], scoring_params)[0]

            full_ids = tokenizer.encode(full_text)
            prompt_ids = tokenizer.encode(prompt)
            n_ans = len(full_ids) - len(prompt_ids)
            if n_ans <= 0:
                n_ans = max(
                    1,
                    len(tokenizer.encode(answer_string, add_special_tokens=False)),
                )

            answer_lp_list: list[float] = []
            prompt_lp = result.prompt_logprobs  # type: ignore[attr-defined]
            if prompt_lp is not None and len(prompt_lp) >= n_ans:
                answer_positions = prompt_lp[-n_ans:]
                for pos_idx, pos_lp in enumerate(answer_positions):
                    if pos_lp is None:
                        continue
                    actual_id = full_ids[len(full_ids) - n_ans + pos_idx]
                    lp_obj = pos_lp.get(actual_id)
                    if lp_obj is None:
                        lp_obj = next(iter(pos_lp.values()))
                    lp_val = getattr(lp_obj, "logprob", lp_obj)
                    answer_lp_list.append(float(lp_val))

            return (
                sum(answer_lp_list) / len(answer_lp_list)
                if answer_lp_list
                else float("nan")
            )

        per_problem: list[dict] = []
        gen_t0 = time.time()

        for i, problem in enumerate(problems):
            n_oracle = int(problem["answer"])
            corrupted_solution, n_corrupted = corrupt_solution_consistent(
                problem["solution"], n_oracle, seed=seed + i
            )

            ans_oracle = f"\\boxed{{{n_oracle}}}"
            ans_corrupted = f"\\boxed{{{n_corrupted}}}"

            prompt_no_cot = build_prompt(problem["problem"], None)
            prompt_oracle = build_prompt(problem["problem"], problem["solution"])
            prompt_consistent = build_prompt(problem["problem"], corrupted_solution)

            logp_no_cot_correct = round(score_answer(prompt_no_cot, ans_oracle), 4)
            logp_no_cot_wrong = round(score_answer(prompt_no_cot, ans_corrupted), 4)

            logp_oracle_correct = round(score_answer(prompt_oracle, ans_oracle), 4)
            logp_oracle_wrong = round(score_answer(prompt_oracle, ans_corrupted), 4)

            logp_corrupted_correct = round(
                score_answer(prompt_consistent, ans_oracle), 4
            )
            logp_corrupted_wrong = round(
                score_answer(prompt_consistent, ans_corrupted), 4
            )

            delta_disc_oracle = round(logp_oracle_correct - logp_oracle_wrong, 4)
            delta_disc_corrupted = round(
                logp_corrupted_correct - logp_corrupted_wrong, 4
            )
            discrimination_signal = round(
                delta_disc_oracle - delta_disc_corrupted, 4
            )

            row = {
                "id": problem.get("id"),
                "year": problem.get("year"),
                "answer": n_oracle,
                "n_corrupted": n_corrupted,
                "logp_no_cot_correct": logp_no_cot_correct,
                "logp_no_cot_wrong": logp_no_cot_wrong,
                "logp_oracle_correct": logp_oracle_correct,
                "logp_oracle_wrong": logp_oracle_wrong,
                "logp_corrupted_correct": logp_corrupted_correct,
                "logp_corrupted_wrong": logp_corrupted_wrong,
                "delta_disc_oracle": delta_disc_oracle,
                "delta_disc_corrupted": delta_disc_corrupted,
                "discrimination_signal": discrimination_signal,
            }
            per_problem.append(row)

        gen_elapsed = time.time() - gen_t0

        with out_path.open("w", encoding="utf-8") as fh:
            for row in per_problem:
                fh.write(json.dumps(_sanitize_row(row), allow_nan=False) + "\n")

        try:
            VOLUMES["/cache"].commit()
        except Exception:  # noqa: BLE001
            pass

        def _mean(key: str) -> float:
            vals = [
                r[key]
                for r in per_problem
                if r.get(key) is not None
                and not (isinstance(r[key], float) and math.isnan(r[key]))
            ]
            return round(sum(vals) / max(1, len(vals)), 4) if vals else float("nan")

        signals = [
            r["discrimination_signal"]
            for r in per_problem
            if r.get("discrimination_signal") is not None
            and not (isinstance(r["discrimination_signal"], float) and math.isnan(r["discrimination_signal"]))
        ]
        pct_oracle_above = round(
            sum(
                1
                for r in per_problem
                if r["delta_disc_oracle"] > r["delta_disc_corrupted"]
            )
            / max(1, len(per_problem)),
            4,
        )
        mean_disc_signal = _mean("discrimination_signal")

        try:
            w_pvalue = float(wilcoxon(signals, alternative="greater").pvalue)
        except Exception:  # noqa: BLE001
            w_pvalue = float("nan")

        gate_passed = (
            mean_disc_signal > 0
            and pct_oracle_above > 0.6
            and w_pvalue < 0.05
        )

        total_elapsed = time.time() - t0
        return {
            "model_id": model_id,
            "n_problems": len(problems),
            "mean_delta_disc_oracle": _mean("delta_disc_oracle"),
            "mean_delta_disc_corrupted": _mean("delta_disc_corrupted"),
            "mean_discrimination_signal": mean_disc_signal,
            "pct_oracle_above_corrupted": pct_oracle_above,
            "wilcoxon_pvalue": round(w_pvalue, 6),
            "gate_passed": gate_passed,
            "gen_elapsed_s": round(gen_elapsed, 2),
            "total_elapsed_s": round(total_elapsed, 2),
            "estimated_cost_usd": estimate_cost("H100", 4, total_elapsed / 3600.0),
            "save_to": str(out_path),
        }

    # -----------------------------------------------------------------------
    # B2) run_random_answer_corruption_logprob (Round-7 reviewer-defense probe)
    # -----------------------------------------------------------------------

    @app.function(
        gpu=gpu_spec("pilot"),
        volumes=VOLUMES,
        secrets=get_secrets(),
        timeout=3600,
        memory=32_000,
    )
    def run_random_answer_corruption_logprob(
        problems: list[dict],
        *,
        model_id: str = "Qwen/Qwen2.5-1.5B",
        seed: int = 1337,
        save_to: str = "/cache/premise/random_answer_corruption_logprob.jsonl",
    ) -> dict:
        """Probe-1 reviewer-defense control: swap the final ``\\boxed{N_oracle}``
        with a digit-matched random in-distribution AIME answer drawn from the
        ground-truth answers of the other problems in the eval set. The chain
        content is left unchanged (no delta-shifted intermediate token), so
        any positive discrimination signal here cannot be explained by an
        out-of-distribution token-frequency artifact of a delta-shifted answer.
        Mirrors the structure of run_consistent_corruption_logprob exactly,
        differing only in the corruption function.
        """
        from scipy.stats import wilcoxon  # noqa: PLC0415
        from vllm import LLM, SamplingParams  # noqa: PLC0415

        out_path = Path(save_to)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Build the in-distribution answer pool from the eval set itself.
        answer_pool = [int(p["answer"]) for p in problems]

        t0 = time.time()
        llm = LLM(
            model=model_id,
            dtype="bfloat16",
            seed=seed,
            tensor_parallel_size=4,
            max_model_len=2048,
            gpu_memory_utilization=0.85,
            trust_remote_code=True,
        )
        tokenizer = llm.get_tokenizer()

        scoring_params = SamplingParams(
            prompt_logprobs=0,
            max_tokens=1,
            temperature=0.0,
            seed=seed,
        )

        def build_prompt(problem: str, cot: str | None) -> str:
            if cot is None or cot == "":
                return (
                    "Solve the following problem. Show your reasoning briefly, "
                    "then put the final integer answer in \\boxed{}.\n\n"
                    f"Problem: {problem}\n\nSolution: "
                )
            return (
                "Solve the following problem. Use the provided solution sketch "
                "as guidance. Put the final integer answer in \\boxed{}.\n\n"
                f"Problem: {problem}\n\nSolution sketch: {cot}\n\n"
                "Final answer: "
            )

        def score_answer(prompt: str, answer_string: str) -> float:
            full_text = prompt + answer_string
            result = llm.generate([full_text], scoring_params)[0]
            full_ids = tokenizer.encode(full_text)
            prompt_ids = tokenizer.encode(prompt)
            n_ans = len(full_ids) - len(prompt_ids)
            if n_ans <= 0:
                n_ans = max(
                    1,
                    len(tokenizer.encode(answer_string, add_special_tokens=False)),
                )
            answer_lp_list: list[float] = []
            prompt_lp = result.prompt_logprobs  # type: ignore[attr-defined]
            if prompt_lp is not None and len(prompt_lp) >= n_ans:
                answer_positions = prompt_lp[-n_ans:]
                for pos_idx, pos_lp in enumerate(answer_positions):
                    if pos_lp is None:
                        continue
                    actual_id = full_ids[len(full_ids) - n_ans + pos_idx]
                    lp_obj = pos_lp.get(actual_id)
                    if lp_obj is None:
                        lp_obj = next(iter(pos_lp.values()))
                    lp_val = getattr(lp_obj, "logprob", lp_obj)
                    answer_lp_list.append(float(lp_val))
            return (
                sum(answer_lp_list) / len(answer_lp_list)
                if answer_lp_list
                else float("nan")
            )

        per_problem: list[dict] = []
        gen_t0 = time.time()

        for i, problem in enumerate(problems):
            n_oracle = int(problem["answer"])
            corrupted_solution, n_random = corrupt_solution_random_answer(
                problem["solution"], n_oracle, answer_pool, seed=seed + i
            )

            ans_oracle = f"\\boxed{{{n_oracle}}}"
            ans_random = f"\\boxed{{{n_random}}}"

            prompt_oracle = build_prompt(problem["problem"], problem["solution"])
            prompt_random = build_prompt(problem["problem"], corrupted_solution)

            logp_oracle_correct = round(score_answer(prompt_oracle, ans_oracle), 4)
            logp_oracle_wrong = round(score_answer(prompt_oracle, ans_random), 4)
            logp_random_correct = round(score_answer(prompt_random, ans_oracle), 4)
            logp_random_wrong = round(score_answer(prompt_random, ans_random), 4)

            delta_disc_oracle = round(logp_oracle_correct - logp_oracle_wrong, 4)
            delta_disc_random = round(
                logp_random_correct - logp_random_wrong, 4
            )
            discrimination_signal = round(
                delta_disc_oracle - delta_disc_random, 4
            )

            row = {
                "id": problem.get("id"),
                "year": problem.get("year"),
                "answer": n_oracle,
                "n_random": n_random,
                "logp_oracle_correct": logp_oracle_correct,
                "logp_oracle_wrong": logp_oracle_wrong,
                "logp_random_correct": logp_random_correct,
                "logp_random_wrong": logp_random_wrong,
                "delta_disc_oracle": delta_disc_oracle,
                "delta_disc_corrupted": delta_disc_random,
                "discrimination_signal": discrimination_signal,
            }
            per_problem.append(row)

        gen_elapsed = time.time() - gen_t0

        with out_path.open("w", encoding="utf-8") as fh:
            for row in per_problem:
                fh.write(json.dumps(_sanitize_row(row), allow_nan=False) + "\n")

        try:
            VOLUMES["/cache"].commit()
        except Exception:  # noqa: BLE001
            pass

        def _mean(key: str) -> float:
            vals = [
                r[key]
                for r in per_problem
                if r.get(key) is not None
                and not (isinstance(r[key], float) and math.isnan(r[key]))
            ]
            return round(sum(vals) / max(1, len(vals)), 4) if vals else float("nan")

        signals = [
            r["discrimination_signal"]
            for r in per_problem
            if r.get("discrimination_signal") is not None
            and not (isinstance(r["discrimination_signal"], float) and math.isnan(r["discrimination_signal"]))
        ]
        pct_oracle_above = round(
            sum(
                1
                for r in per_problem
                if r["delta_disc_oracle"] > r["delta_disc_corrupted"]
            )
            / max(1, len(per_problem)),
            4,
        )
        mean_disc_signal = _mean("discrimination_signal")

        try:
            w_pvalue = float(wilcoxon(signals, alternative="greater").pvalue)
        except Exception:  # noqa: BLE001
            w_pvalue = float("nan")

        gate_passed = (
            mean_disc_signal > 0
            and pct_oracle_above > 0.6
            and w_pvalue < 0.05
        )

        total_elapsed = time.time() - t0
        return {
            "model_id": model_id,
            "n_problems": len(problems),
            "mean_delta_disc_oracle": _mean("delta_disc_oracle"),
            "mean_delta_disc_corrupted": _mean("delta_disc_corrupted"),
            "mean_discrimination_signal": mean_disc_signal,
            "pct_oracle_above_corrupted": pct_oracle_above,
            "wilcoxon_pvalue": round(w_pvalue, 6),
            "gate_passed": gate_passed,
            "gen_elapsed_s": round(gen_elapsed, 2),
            "total_elapsed_s": round(total_elapsed, 2),
            "estimated_cost_usd": estimate_cost("H100", 4, total_elapsed / 3600.0),
            "save_to": str(out_path),
        }

    # -----------------------------------------------------------------------
    # C) run_verdict_test
    # -----------------------------------------------------------------------

    @app.function(
        gpu=gpu_spec("pilot"),
        volumes=VOLUMES,
        secrets=get_secrets(),
        timeout=3600,
        memory=32_000,
    )
    def run_verdict_test(
        problems: list[dict],
        *,
        model_id: str = "Qwen/Qwen2.5-1.5B",
        seed: int = 1337,
        save_to: str = "/cache/premise/verdict_test.jsonl",
    ) -> dict:
        """Instruction-following verdict grader.

        For each problem we build a YES/NO grading prompt under three
        conditions (oracle, consistent_corrupted, shuffled) and score
        logp(" YES") and logp(" NO") (with leading space). The verdict
        logit is logp_yes - logp_no. A higher logit for the oracle
        condition shows the model can discriminate correct solutions.

        Gate: mean_delta_verdict > 0
              AND pct_oracle_above_corrupted > 0.6
              AND Wilcoxon one-sided p < 0.05
        """
        from scipy.stats import wilcoxon  # noqa: PLC0415
        from vllm import LLM, SamplingParams  # noqa: PLC0415

        out_path = Path(save_to)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        llm = LLM(
            model=model_id,
            dtype="bfloat16",
            seed=seed,
            tensor_parallel_size=4,
            max_model_len=2048,
            gpu_memory_utilization=0.85,
            trust_remote_code=True,
        )
        tokenizer = llm.get_tokenizer()

        scoring_params = SamplingParams(
            prompt_logprobs=0,
            max_tokens=1,
            temperature=0.0,
            seed=seed,
        )

        _VERDICT_TEMPLATE = (
            "You are a careful math grader. Read the problem and the proposed "
            "solution, then judge whether the proposed solution is correct.\n\n"
            "Problem: {problem}\n\n"
            "Proposed solution: {solution}\n\n"
            "Is the proposed solution correct? Answer YES or NO.\n\n"
            "Answer:"
        )

        def build_verdict_prompt(problem: str, solution: str) -> str:
            return _VERDICT_TEMPLATE.format(problem=problem, solution=solution)

        def score_verdict_token(prompt: str, token_str: str) -> float:
            """Score logp of a single verdict token (e.g. ' YES' or ' NO')."""
            full_text = prompt + token_str
            result = llm.generate([full_text], scoring_params)[0]

            full_ids = tokenizer.encode(full_text)
            prompt_ids = tokenizer.encode(prompt)
            n_ans = len(full_ids) - len(prompt_ids)
            if n_ans <= 0:
                n_ans = max(
                    1,
                    len(tokenizer.encode(token_str, add_special_tokens=False)),
                )

            answer_lp_list: list[float] = []
            prompt_lp = result.prompt_logprobs  # type: ignore[attr-defined]
            if prompt_lp is not None and len(prompt_lp) >= n_ans:
                answer_positions = prompt_lp[-n_ans:]
                for pos_idx, pos_lp in enumerate(answer_positions):
                    if pos_lp is None:
                        continue
                    actual_id = full_ids[len(full_ids) - n_ans + pos_idx]
                    lp_obj = pos_lp.get(actual_id)
                    if lp_obj is None:
                        lp_obj = next(iter(pos_lp.values()))
                    lp_val = getattr(lp_obj, "logprob", lp_obj)
                    answer_lp_list.append(float(lp_val))

            return (
                sum(answer_lp_list) / len(answer_lp_list)
                if answer_lp_list
                else float("nan")
            )

        def verdict_logit(prompt: str) -> float:
            logp_yes = score_verdict_token(prompt, " YES")
            logp_no = score_verdict_token(prompt, " NO")
            return round(logp_yes - logp_no, 4)

        per_problem: list[dict] = []
        gen_t0 = time.time()

        for i, problem in enumerate(problems):
            n_oracle = int(problem["answer"])
            corrupted_solution, _n_corrupted = corrupt_solution_consistent(
                problem["solution"], n_oracle, seed=seed + i
            )
            shuffled_solution = shuffle_solution_steps(
                problem["solution"], seed=seed + i
            )

            prompt_oracle = build_verdict_prompt(
                problem["problem"], problem["solution"]
            )
            prompt_corrupted = build_verdict_prompt(
                problem["problem"], corrupted_solution
            )
            prompt_shuffled = build_verdict_prompt(
                problem["problem"], shuffled_solution
            )

            logit_oracle = verdict_logit(prompt_oracle)
            logit_corrupted = verdict_logit(prompt_corrupted)
            logit_shuffled = verdict_logit(prompt_shuffled)

            row = {
                "id": problem.get("id"),
                "year": problem.get("year"),
                "answer": n_oracle,
                "logit_oracle": logit_oracle,
                "logit_corrupted": logit_corrupted,
                "logit_shuffled": logit_shuffled,
                "delta_oracle_minus_corrupted": round(
                    logit_oracle - logit_corrupted, 4
                ),
            }
            per_problem.append(row)

        gen_elapsed = time.time() - gen_t0

        with out_path.open("w", encoding="utf-8") as fh:
            for row in per_problem:
                fh.write(json.dumps(_sanitize_row(row), allow_nan=False) + "\n")

        try:
            VOLUMES["/cache"].commit()
        except Exception:  # noqa: BLE001
            pass

        def _mean(key: str) -> float:
            vals = [
                r[key]
                for r in per_problem
                if r.get(key) is not None
                and not (isinstance(r[key], float) and math.isnan(r[key]))
            ]
            return round(sum(vals) / max(1, len(vals)), 4) if vals else float("nan")

        deltas = [
            r["delta_oracle_minus_corrupted"]
            for r in per_problem
            if r.get("delta_oracle_minus_corrupted") is not None
            and not (isinstance(r["delta_oracle_minus_corrupted"], float) and math.isnan(r["delta_oracle_minus_corrupted"]))
        ]
        pct_oracle_above = round(
            sum(
                1
                for r in per_problem
                if r["logit_oracle"] > r["logit_corrupted"]
            )
            / max(1, len(per_problem)),
            4,
        )
        mean_delta = _mean("delta_oracle_minus_corrupted")

        try:
            w_pvalue = float(wilcoxon(deltas, alternative="greater").pvalue)
        except Exception:  # noqa: BLE001
            w_pvalue = float("nan")

        gate_passed = (
            mean_delta > 0
            and pct_oracle_above > 0.6
            and w_pvalue < 0.05
        )

        total_elapsed = time.time() - t0
        return {
            "model_id": model_id,
            "n_problems": len(problems),
            "mean_logit_oracle": _mean("logit_oracle"),
            "mean_logit_corrupted": _mean("logit_corrupted"),
            "mean_logit_shuffled": _mean("logit_shuffled"),
            "mean_delta_verdict": mean_delta,
            "pct_oracle_above_corrupted": pct_oracle_above,
            "wilcoxon_pvalue": round(w_pvalue, 6),
            "gate_passed": gate_passed,
            "gen_elapsed_s": round(gen_elapsed, 2),
            "total_elapsed_s": round(total_elapsed, 2),
            "estimated_cost_usd": estimate_cost("H100", 4, total_elapsed / 3600.0),
            "save_to": str(out_path),
        }

    # -----------------------------------------------------------------------
    # D) run_format_ablation_logprob
    # -----------------------------------------------------------------------

    @app.function(
        gpu=gpu_spec("pilot"),
        volumes=VOLUMES,
        secrets=get_secrets(),
        timeout=3600,
        memory=32_000,
    )
    def run_format_ablation_logprob(
        problems: list[dict],
        *,
        model_id: str = "Qwen/Qwen2.5-1.5B",
        template: str = "bare",
        seed: int = 1337,
        save_to: str = "/cache/premise/format_ablation_logprob.jsonl",
    ) -> dict:
        """Run the logprob premise test under different prompt templates.

        ``template`` must be one of ``"bare"``, ``"chat"``, or
        ``"interleaved"``.

        - ``bare``: matches the original ``run_logprob_premise_test`` format
          (sanity baseline).
        - ``chat``: uses ``tokenizer.apply_chat_template``; falls back to
          ``bare`` per-problem if the tokenizer has no chat template.
        - ``interleaved``: wraps problem and CoT in XML-like tags.

        Four conditions per problem: no_cot, oracle, corrupted, shuffled.
        Corruption uses ``corrupt_solution_one_step`` (original, not
        consistent variant). Scores ``\\boxed{N}`` against ground-truth
        answer.

        Aggregate output matches ``run_logprob_premise_test`` plus the
        keys ``"template"`` and ``"n_chat_fallback"``.
        """
        from scipy.stats import wilcoxon  # noqa: PLC0415
        from vllm import LLM, SamplingParams  # noqa: PLC0415

        if template not in ("bare", "chat", "interleaved"):
            raise ValueError(
                f"template must be 'bare', 'chat', or 'interleaved'; got {template!r}"
            )

        out_path = Path(save_to)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        llm = LLM(
            model=model_id,
            dtype="bfloat16",
            seed=seed,
            tensor_parallel_size=4,
            max_model_len=2048,
            gpu_memory_utilization=0.85,
            trust_remote_code=True,
        )
        tokenizer = llm.get_tokenizer()

        scoring_params = SamplingParams(
            prompt_logprobs=0,
            max_tokens=1,
            temperature=0.0,
            seed=seed,
        )

        n_chat_fallback: int = 0

        def _bare_prompt(problem: str, cot: str | None) -> str:
            if cot is None or cot == "":
                return (
                    "Solve the following problem. Show your reasoning briefly, "
                    "then put the final integer answer in \\boxed{}.\n\n"
                    f"Problem: {problem}\n\nSolution: "
                )
            return (
                "Solve the following problem. Use the provided solution sketch "
                "as guidance. Put the final integer answer in \\boxed{}.\n\n"
                f"Problem: {problem}\n\nSolution sketch: {cot}\n\n"
                "Final answer: "
            )

        def _chat_content(problem: str, cot: str | None) -> str:
            if cot is None or cot == "":
                return (
                    "Solve the following problem. Show your reasoning briefly, "
                    "then put the final integer answer in \\boxed{}.\n\n"
                    f"Problem: {problem}"
                )
            return (
                "Solve the following problem. Use the provided solution sketch "
                "as guidance. Put the final integer answer in \\boxed{}.\n\n"
                f"Problem: {problem}\n\nSolution sketch: {cot}"
            )

        def _interleaved_prompt(problem: str, cot: str | None) -> str:
            if cot is None or cot == "":
                return f"<problem>\n{problem}\n</problem>\n<answer>\n"
            return (
                f"<problem>\n{problem}\n</problem>\n"
                f"<solution_sketch>\n{cot}\n</solution_sketch>\n"
                "<answer>\n"
            )

        def build_prompt(problem: str, cot: str | None) -> str:
            nonlocal n_chat_fallback
            if template == "bare":
                return _bare_prompt(problem, cot)
            elif template == "chat":
                content = _chat_content(problem, cot)
                messages = [{"role": "user", "content": content}]
                try:
                    return tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:  # noqa: BLE001
                    n_chat_fallback += 1
                    return _bare_prompt(problem, cot)
            else:  # interleaved
                return _interleaved_prompt(problem, cot)

        per_problem: list[dict] = []
        gen_t0 = time.time()

        for i, problem in enumerate(problems):
            answer_string = f"\\boxed{{{int(problem['answer'])}}}"
            prompts = {
                "no_cot": build_prompt(problem["problem"], None),
                "oracle": build_prompt(problem["problem"], problem["solution"]),
                "corrupted": build_prompt(
                    problem["problem"],
                    corrupt_solution_one_step(problem["solution"], seed=seed + i),
                ),
                "shuffled": build_prompt(
                    problem["problem"],
                    shuffle_solution_steps(problem["solution"], seed=seed + i),
                ),
            }

            logps: dict[str, float] = {}
            n_answer_tokens: int | None = None

            for cond, prompt in prompts.items():
                full_text = prompt + answer_string
                result = llm.generate([full_text], scoring_params)[0]

                full_ids = tokenizer.encode(full_text)
                prompt_ids = tokenizer.encode(prompt)
                n_ans = len(full_ids) - len(prompt_ids)
                if n_ans <= 0:
                    n_ans = max(
                        1,
                        len(
                            tokenizer.encode(answer_string, add_special_tokens=False)
                        ),
                    )

                answer_lp_list: list[float] = []
                prompt_lp = result.prompt_logprobs  # type: ignore[attr-defined]
                if prompt_lp is not None and len(prompt_lp) >= n_ans:
                    answer_positions = prompt_lp[-n_ans:]
                    for pos_idx, pos_lp in enumerate(answer_positions):
                        if pos_lp is None:
                            continue
                        actual_id = full_ids[len(full_ids) - n_ans + pos_idx]
                        lp_obj = pos_lp.get(actual_id)
                        if lp_obj is None:
                            lp_obj = next(iter(pos_lp.values()))
                        lp_val = getattr(lp_obj, "logprob", lp_obj)
                        answer_lp_list.append(float(lp_val))

                mean_lp = (
                    sum(answer_lp_list) / len(answer_lp_list)
                    if answer_lp_list
                    else float("nan")
                )
                logps[cond] = round(mean_lp, 4)
                if n_answer_tokens is None:
                    n_answer_tokens = len(answer_lp_list)

            delta_oc = round(logps["oracle"] - logps["corrupted"], 4)
            row = {
                "id": problem.get("id"),
                "year": problem.get("year"),
                "answer": problem["answer"],
                "answer_string": answer_string,
                "n_answer_tokens": n_answer_tokens,
                "logp_no_cot": logps["no_cot"],
                "logp_oracle": logps["oracle"],
                "logp_corrupted": logps["corrupted"],
                "logp_shuffled": logps["shuffled"],
                "delta_oracle_corrupted": delta_oc,
                "oracle_beats_corrupted": delta_oc > 0,
            }
            per_problem.append(row)

        gen_elapsed = time.time() - gen_t0

        with out_path.open("w", encoding="utf-8") as fh:
            for row in per_problem:
                fh.write(json.dumps(_sanitize_row(row), allow_nan=False) + "\n")

        try:
            VOLUMES["/cache"].commit()
        except Exception:  # noqa: BLE001
            pass

        def _mean(key: str) -> float:
            vals = [
                r[key]
                for r in per_problem
                if r.get(key) is not None
                and not (isinstance(r[key], float) and math.isnan(r[key]))
            ]
            return round(sum(vals) / max(1, len(vals)), 4) if vals else float("nan")

        deltas = [
            r["delta_oracle_corrupted"]
            for r in per_problem
            if r.get("delta_oracle_corrupted") is not None
            and not (isinstance(r["delta_oracle_corrupted"], float) and math.isnan(r["delta_oracle_corrupted"]))
        ]
        pct_beats = round(
            sum(1 for r in per_problem if r["oracle_beats_corrupted"])
            / max(1, len(per_problem)),
            4,
        )
        mean_delta = _mean("delta_oracle_corrupted")

        try:
            w_pvalue = float(wilcoxon(deltas, alternative="greater").pvalue)
        except Exception:  # noqa: BLE001
            w_pvalue = float("nan")

        gate_passed = mean_delta > 0 and pct_beats > 0.6 and w_pvalue < 0.05

        total_elapsed = time.time() - t0
        return {
            "model_id": model_id,
            "n_problems": len(problems),
            "template": template,
            "n_chat_fallback": n_chat_fallback,
            "mean_logp_no_cot": _mean("logp_no_cot"),
            "mean_logp_oracle": _mean("logp_oracle"),
            "mean_logp_corrupted": _mean("logp_corrupted"),
            "mean_logp_shuffled": _mean("logp_shuffled"),
            "delta_oracle_minus_corrupted": mean_delta,
            "pct_oracle_beats_corrupted": pct_beats,
            "wilcoxon_pvalue": round(w_pvalue, 6),
            "gate_logprob_passed": gate_passed,
            "gen_elapsed_s": round(gen_elapsed, 2),
            "total_elapsed_s": round(total_elapsed, 2),
            "estimated_cost_usd": estimate_cost("H100", 4, total_elapsed / 3600.0),
            "save_to": str(out_path),
        }

    # -----------------------------------------------------------------------
    # E) Local entrypoints
    # -----------------------------------------------------------------------

    @app.local_entrypoint()
    def main_consistent(
        problems_jsonl: str,
        *,
        model_id: str = "Qwen/Qwen2.5-1.5B",
        seed: int = 1337,
        save_to: str = "/cache/premise/consistent_corruption_logprob.jsonl",
    ) -> None:
        path = Path(problems_jsonl)
        if not path.exists():
            raise FileNotFoundError(
                f"problems file not found: {path}. Run `python "
                "scripts/fetch_data.py` first."
            )
        problems = [
            json.loads(l)
            for l in path.read_text().splitlines()
            if l.strip()
        ]
        problems = [p for p in problems if p.get("solution")]
        print(
            f"[consistent_corruption] {len(problems)} problems → 3 conditions "
            f"× 2 answers (log-prob scoring) on {model_id}"
        )
        summary = run_consistent_corruption_logprob.remote(
            problems,
            model_id=model_id,
            seed=seed,
            save_to=save_to,
        )
        print("[consistent_corruption] summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        gate = summary["gate_passed"]
        disc = summary["mean_discrimination_signal"]
        pct = summary["pct_oracle_above_corrupted"]
        pval = summary["wilcoxon_pvalue"]
        print(
            f"\n[gate (consistent)] discrimination_signal = {disc:.4f}; "
            f"pct_oracle_above = {pct:.3f}; Wilcoxon p = {pval:.4f}; "
            f"PASSED={gate}"
        )
        if not gate:
            print(
                "[gate (consistent)] FAILED — consistent-corruption "
                "discrimination signal absent."
            )

    @app.local_entrypoint()
    def main_random_answer(
        problems_jsonl: str,
        *,
        model_id: str = "Qwen/Qwen2.5-1.5B",
        seed: int = 1337,
        save_to: str = "/cache/premise/random_answer_corruption.jsonl",
    ) -> None:
        path = Path(problems_jsonl)
        if not path.exists():
            raise FileNotFoundError(
                f"problems file not found: {path}. Run `python "
                "scripts/fetch_data.py` first."
            )
        problems = [
            json.loads(l)
            for l in path.read_text().splitlines()
            if l.strip()
        ]
        problems = [p for p in problems if p.get("solution")]
        print(
            f"[random_answer_corruption] {len(problems)} problems → "
            f"oracle vs random-answer-swap (in-distribution AIME pool) on {model_id}"
        )
        summary = run_random_answer_corruption_logprob.remote(
            problems,
            model_id=model_id,
            seed=seed,
            save_to=save_to,
        )
        print("[random_answer_corruption] summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        gate = summary["gate_passed"]
        disc = summary["mean_discrimination_signal"]
        pct = summary["pct_oracle_above_corrupted"]
        pval = summary["wilcoxon_pvalue"]
        print(
            f"\n[gate (random_answer)] discrimination_signal = {disc:.4f}; "
            f"pct_oracle_above = {pct:.3f}; Wilcoxon p = {pval:.4f}; "
            f"PASSED={gate}"
        )

    @app.local_entrypoint()
    def main_verdict(
        problems_jsonl: str,
        *,
        model_id: str = "Qwen/Qwen2.5-1.5B",
        seed: int = 1337,
        save_to: str = "/cache/premise/verdict_test.jsonl",
    ) -> None:
        path = Path(problems_jsonl)
        if not path.exists():
            raise FileNotFoundError(
                f"problems file not found: {path}. Run `python "
                "scripts/fetch_data.py` first."
            )
        problems = [
            json.loads(l)
            for l in path.read_text().splitlines()
            if l.strip()
        ]
        problems = [p for p in problems if p.get("solution")]
        print(
            f"[verdict_test] {len(problems)} problems → 3 conditions "
            f"(YES/NO verdict logit) on {model_id}"
        )
        summary = run_verdict_test.remote(
            problems,
            model_id=model_id,
            seed=seed,
            save_to=save_to,
        )
        print("[verdict_test] summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        gate = summary["gate_passed"]
        delta = summary["mean_delta_verdict"]
        pct = summary["pct_oracle_above_corrupted"]
        pval = summary["wilcoxon_pvalue"]
        print(
            f"\n[gate (verdict)] mean_delta_verdict = {delta:.4f}; "
            f"pct_oracle_above = {pct:.3f}; Wilcoxon p = {pval:.4f}; "
            f"PASSED={gate}"
        )
        if not gate:
            print(
                "[gate (verdict)] FAILED — verdict logit does not discriminate "
                "oracle from consistently-corrupted solutions."
            )

    @app.local_entrypoint()
    def main_format(
        problems_jsonl: str,
        *,
        model_id: str = "Qwen/Qwen2.5-1.5B",
        template: str = "bare",
        save_to: str = "/cache/premise/format_ablation_logprob.jsonl",
    ) -> None:
        path = Path(problems_jsonl)
        if not path.exists():
            raise FileNotFoundError(
                f"problems file not found: {path}. Run `python "
                "scripts/fetch_data.py` first."
            )
        problems = [
            json.loads(l)
            for l in path.read_text().splitlines()
            if l.strip()
        ]
        problems = [p for p in problems if p.get("solution")]
        print(
            f"[format_ablation] {len(problems)} problems → 4 conditions "
            f"(template={template!r}) on {model_id}"
        )
        summary = run_format_ablation_logprob.remote(
            problems,
            model_id=model_id,
            template=template,
            save_to=save_to,
        )
        print("[format_ablation] summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        gate = summary["gate_logprob_passed"]
        delta = summary["delta_oracle_minus_corrupted"]
        pct = summary["pct_oracle_beats_corrupted"]
        pval = summary["wilcoxon_pvalue"]
        n_fb = summary["n_chat_fallback"]
        print(
            f"\n[gate (format/{template})] Δ(oracle−corrupted) = {delta:.4f}; "
            f"pct_oracle_beats = {pct:.3f}; Wilcoxon p = {pval:.4f}; "
            f"n_chat_fallback = {n_fb}; PASSED={gate}"
        )
        if not gate:
            print(
                f"[gate (format/{template})] FAILED — log-prob discrimination "
                "absent under this template."
            )

    # -----------------------------------------------------------------------
    # F) run_per_token_surprise
    # -----------------------------------------------------------------------

    WINDOW_RADIUS = 3

    @app.function(
        gpu=gpu_spec("pilot"),
        volumes=VOLUMES,
        secrets=get_secrets(),
        timeout=3600,
        memory=32_000,
    )
    def run_per_token_surprise(
        problems: list[dict],
        *,
        model_id: str = "Qwen/Qwen2.5-1.5B",
        seed: int = 1337,
        save_to: str = "/cache/premise/per_token_surprise.jsonl",
    ) -> dict:
        """Per-token surprise (negative logprob) mechanistic probe.

        For each AIME problem, computes per-token surprise along the
        chain-of-thought tokens under (a) oracle and (b) consistent_corrupted
        conditions. Detects whether the model is locally surprised at the
        position of the corruption even if that surprise does not propagate
        to the final answer logprob.

        Aggregate gate:
          mean_local_minus_global > 0
          AND pct_positive_local_minus_global > 0.6
          AND Wilcoxon one-sided p < 0.05
        """
        from scipy.stats import wilcoxon  # noqa: PLC0415
        from vllm import LLM, SamplingParams  # noqa: PLC0415

        _WINDOW_RADIUS = 3  # ±3 tokens around each corruption site

        out_path = Path(save_to)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        llm = LLM(
            model=model_id,
            dtype="bfloat16",
            seed=seed,
            tensor_parallel_size=4,
            max_model_len=2048,
            gpu_memory_utilization=0.85,
            trust_remote_code=True,
        )
        tokenizer = llm.get_tokenizer()

        scoring_params = SamplingParams(
            prompt_logprobs=0,
            max_tokens=1,
            temperature=0.0,
            seed=seed,
        )

        def build_prompt(problem_text: str, cot: str) -> str:
            return (
                "Solve the following problem. Use the provided solution sketch "
                "as guidance. Put the final integer answer in \\boxed{}.\n\n"
                f"Problem: {problem_text}\n\nSolution sketch: {cot}\n\n"
                "Final answer: "
            )

        def get_per_token_surprises(prompt: str) -> list[float | None]:
            """Return per-token surprise (−logp) for each prompt token position.

            Position 0 is always None (vLLM returns None for the first token).
            Returns a list of length == len(tokenized prompt), with None where
            the actual-token logprob is unavailable.
            """
            result = llm.generate([prompt], scoring_params)[0]
            token_ids = tokenizer.encode(prompt)
            prompt_lp = result.prompt_logprobs  # type: ignore[attr-defined]

            surprises: list[float | None] = []
            if prompt_lp is None:
                return [None] * len(token_ids)

            for pos_idx, pos_lp in enumerate(prompt_lp):
                if pos_lp is None:
                    surprises.append(None)
                    continue
                actual_id = token_ids[pos_idx]
                lp_obj = pos_lp.get(actual_id)
                if lp_obj is None:
                    # Mirror the existing reference scoring path: with
                    # prompt_logprobs=0 vLLM returns the actual token's logprob,
                    # so any value() entry is correct (defensive fallback).
                    if pos_lp:
                        lp_obj = next(iter(pos_lp.values()))
                    else:
                        surprises.append(None)
                        continue
                lp_val = float(getattr(lp_obj, "logprob", lp_obj))
                surprises.append(-lp_val)  # surprise = -logp

            # Pad or trim to match token_ids length (defensive).
            while len(surprises) < len(token_ids):
                surprises.append(None)
            return surprises[: len(token_ids)]

        def _mean_of(values: list[float | None]) -> float | None:
            valid = [v for v in values if v is not None]
            if not valid:
                return None
            return sum(valid) / len(valid)

        def _window_indices(center: int, radius: int, max_idx: int) -> list[int]:
            return [
                i
                for i in range(center - radius, center + radius + 1)
                if 0 <= i <= max_idx
            ]

        per_problem: list[dict] = []
        gen_t0 = time.time()

        for prob_i, problem in enumerate(problems):
            n_oracle = int(problem["answer"])
            corrupted_solution, n_corrupted = corrupt_solution_consistent(
                problem["solution"], n_oracle, seed=seed + prob_i
            )

            prompt_oracle = build_prompt(problem["problem"], problem["solution"])
            prompt_corrupted = build_prompt(problem["problem"], corrupted_solution)

            oracle_ids = tokenizer.encode(prompt_oracle)
            corrupted_ids = tokenizer.encode(prompt_corrupted)

            len_oracle = len(oracle_ids)
            len_corrupted = len(corrupted_ids)

            # Identify chain-of-thought token count (everything in the prompt
            # after the preamble; approximate as the full prompt for per-token
            # surprise purposes — we want ALL prompt positions).
            n_oracle_chain_tokens = len_oracle
            n_corrupted_chain_tokens = len_corrupted

            # Find first and last differing token indices.
            min_len = min(len_oracle, len_corrupted)
            first_diff = None
            for k in range(min_len):
                if oracle_ids[k] != corrupted_ids[k]:
                    first_diff = k
                    break

            last_diff = None
            for k in range(min_len - 1, -1, -1):
                if oracle_ids[k] != corrupted_ids[k]:
                    last_diff = k
                    break

            # If sequences are identical up to min_len but differ in length,
            # mark the split point as the first/last diff.
            if first_diff is None and len_oracle != len_corrupted:
                first_diff = min_len
                last_diff = min_len
            # If lengths differ AND a diff was found in the shared prefix,
            # the tail (beyond min_len) also contains a divergence — extend
            # last_diff to include the length-change point so the window
            # covers the final-boxed corruption site.
            elif first_diff is not None and len_oracle != len_corrupted:
                last_diff = max(last_diff if last_diff is not None else 0, min_len)

            if first_diff is None:
                # Sequences are identical — nothing to measure; skip.
                row = {
                    "id": problem.get("id"),
                    "year": problem.get("year"),
                    "answer": n_oracle,
                    "n_corrupted": n_corrupted,
                    "n_oracle_chain_tokens": n_oracle_chain_tokens,
                    "n_corrupted_chain_tokens": n_corrupted_chain_tokens,
                    "first_diff_idx": None,
                    "last_diff_idx": None,
                    "skipped": True,
                    "skip_reason": "sequences identical after corruption",
                }
                per_problem.append(_sanitize_row(row))
                continue

            # Guard against large length mismatches.
            if abs(len_oracle - len_corrupted) > 6:
                row = {
                    "id": problem.get("id"),
                    "year": problem.get("year"),
                    "answer": n_oracle,
                    "n_corrupted": n_corrupted,
                    "n_oracle_chain_tokens": n_oracle_chain_tokens,
                    "n_corrupted_chain_tokens": n_corrupted_chain_tokens,
                    "first_diff_idx": first_diff,
                    "last_diff_idx": last_diff,
                    "skipped": True,
                    "skip_reason": (
                        f"length mismatch {len_oracle} vs {len_corrupted} > 6"
                    ),
                }
                per_problem.append(_sanitize_row(row))
                continue

            # Collect per-token surprises for both prompts.
            surp_oracle = get_per_token_surprises(prompt_oracle)
            surp_corrupted = get_per_token_surprises(prompt_corrupted)

            # Build corruption window (union of ±WINDOW_RADIUS around first_diff
            # and last_diff) clamped to corrupted prompt bounds.
            max_idx_corrupted = len_corrupted - 1
            max_idx_oracle = len_oracle - 1

            window_set: set[int] = set()
            for center in ({first_diff, last_diff} if first_diff != last_diff else {first_diff}):
                for idx in _window_indices(center, _WINDOW_RADIUS, max_idx_corrupted):
                    window_set.add(idx)

            # Oracle window: build independently around the same diff centers,
            # clamped to oracle length. Do NOT clamp every corrupted-window
            # index — that produces systematic bias (oracle's last token gets
            # repeatedly counted whenever len_oracle < len_corrupted).
            window_set_oracle: set[int] = set()
            centers = ({first_diff, last_diff} if first_diff != last_diff else {first_diff})
            for center in centers:
                # Clamp center to oracle bounds first (protects against centers
                # that point at positions only present in the longer prompt).
                center_oracle = min(center, max_idx_oracle)
                for idx in _window_indices(center_oracle, _WINDOW_RADIUS, max_idx_oracle):
                    window_set_oracle.add(idx)

            # Collect surprises at window positions.
            surp_corrupted_window = [
                surp_corrupted[i] for i in sorted(window_set) if i < len(surp_corrupted)
            ]
            surp_oracle_window = [
                surp_oracle[i] for i in sorted(window_set_oracle) if i < len(surp_oracle)
            ]

            # Collect surprises at non-window positions (all other positions).
            surp_corrupted_else = [
                surp_corrupted[i]
                for i in range(len_corrupted)
                if i not in window_set
            ]
            surp_oracle_else = [
                surp_oracle[i]
                for i in range(len_oracle)
                if i not in window_set_oracle
            ]

            mean_surp_oracle_window = _mean_of(surp_oracle_window)
            mean_surp_corrupted_window = _mean_of(surp_corrupted_window)
            mean_surp_oracle_else = _mean_of(surp_oracle_else)
            mean_surp_corrupted_else = _mean_of(surp_corrupted_else)

            def _safe_delta(a: float | None, b: float | None) -> float | None:
                if a is None or b is None:
                    return None
                return round(a - b, 4)

            delta_at_window = _safe_delta(
                mean_surp_corrupted_window, mean_surp_oracle_window
            )
            delta_elsewhere = _safe_delta(
                mean_surp_corrupted_else, mean_surp_oracle_else
            )
            local_minus_global = _safe_delta(delta_at_window, delta_elsewhere)

            n_window = len(window_set)

            row = {
                "id": problem.get("id"),
                "year": problem.get("year"),
                "answer": n_oracle,
                "n_corrupted": n_corrupted,
                "n_oracle_chain_tokens": n_oracle_chain_tokens,
                "n_corrupted_chain_tokens": n_corrupted_chain_tokens,
                "first_diff_idx": first_diff,
                "last_diff_idx": last_diff,
                "n_corruption_window_tokens": n_window,
                "mean_surprise_oracle_at_window": (
                    round(mean_surp_oracle_window, 4)
                    if mean_surp_oracle_window is not None
                    else None
                ),
                "mean_surprise_corrupted_at_window": (
                    round(mean_surp_corrupted_window, 4)
                    if mean_surp_corrupted_window is not None
                    else None
                ),
                "mean_surprise_oracle_elsewhere": (
                    round(mean_surp_oracle_else, 4)
                    if mean_surp_oracle_else is not None
                    else None
                ),
                "mean_surprise_corrupted_elsewhere": (
                    round(mean_surp_corrupted_else, 4)
                    if mean_surp_corrupted_else is not None
                    else None
                ),
                "delta_at_window": delta_at_window,
                "delta_elsewhere": delta_elsewhere,
                "local_minus_global": local_minus_global,
                "skipped": False,
            }
            per_problem.append(_sanitize_row(row))

        gen_elapsed = time.time() - gen_t0

        with out_path.open("w", encoding="utf-8") as fh:
            for row in per_problem:
                fh.write(json.dumps(row, allow_nan=False) + "\n")

        try:
            VOLUMES["/cache"].commit()
        except Exception:  # noqa: BLE001
            pass

        # Aggregate over non-skipped problems.
        valid_rows = [r for r in per_problem if not r.get("skipped")]

        lmg_vals = [
            r["local_minus_global"]
            for r in valid_rows
            if r.get("local_minus_global") is not None
            and not (
                isinstance(r["local_minus_global"], float)
                and math.isnan(r["local_minus_global"])
            )
        ]

        def _mean_agg(vals: list[float]) -> float:
            return round(sum(vals) / max(1, len(vals)), 4) if vals else float("nan")

        mean_lmg = _mean_agg(lmg_vals)
        pct_positive_lmg = round(
            sum(1 for v in lmg_vals if v > 0) / max(1, len(lmg_vals)), 4
        )

        try:
            w_pvalue = float(wilcoxon(lmg_vals, alternative="greater").pvalue)
        except Exception:  # noqa: BLE001
            w_pvalue = float("nan")

        gate_passed = (
            mean_lmg > 0
            and pct_positive_lmg > 0.6
            and w_pvalue < 0.05
        )

        total_elapsed = time.time() - t0
        return {
            "model_id": model_id,
            "n_problems": len(problems),
            "n_skipped": len(per_problem) - len(valid_rows),
            "mean_local_minus_global": mean_lmg,
            "pct_positive_local_minus_global": pct_positive_lmg,
            "wilcoxon_pvalue": round(w_pvalue, 6),
            "gate_passed": gate_passed,
            "gen_elapsed_s": round(gen_elapsed, 2),
            "total_elapsed_s": round(total_elapsed, 2),
            "estimated_cost_usd": estimate_cost("H100", 4, total_elapsed / 3600.0),
            "save_to": str(out_path),
        }

    # -----------------------------------------------------------------------
    # G) Local entrypoint for per-token surprise
    # -----------------------------------------------------------------------

    @app.local_entrypoint()
    def main_surprise(
        problems_jsonl: str,
        *,
        model_id: str = "Qwen/Qwen2.5-1.5B",
        seed: int = 1337,
        save_to: str = "/cache/premise/per_token_surprise.jsonl",
    ) -> None:
        path = Path(problems_jsonl)
        if not path.exists():
            raise FileNotFoundError(
                f"problems file not found: {path}. Run `python "
                "scripts/fetch_data.py` first."
            )
        problems = [
            json.loads(l)
            for l in path.read_text().splitlines()
            if l.strip()
        ]
        problems = [p for p in problems if p.get("solution")]
        print(
            f"[per_token_surprise] {len(problems)} problems → oracle + "
            f"consistent_corrupted per-token surprise on {model_id}"
        )
        summary = run_per_token_surprise.remote(
            problems,
            model_id=model_id,
            seed=seed,
            save_to=save_to,
        )
        print("[per_token_surprise] summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        gate = summary["gate_passed"]
        lmg = summary["mean_local_minus_global"]
        pct = summary["pct_positive_local_minus_global"]
        pval = summary["wilcoxon_pvalue"]
        n_skip = summary["n_skipped"]
        print(
            f"\n[gate (per_token_surprise)] mean_local_minus_global = {lmg:.4f}; "
            f"pct_positive = {pct:.3f}; Wilcoxon p = {pval:.4f}; "
            f"n_skipped = {n_skip}; PASSED={gate}"
        )
        if gate:
            print(
                "[gate (per_token_surprise)] PASSED — model shows elevated "
                "local surprise at corruption sites (localized discrimination)."
            )
        else:
            print(
                "[gate (per_token_surprise)] FAILED — no significant local "
                "surprise elevation at corruption sites."
            )

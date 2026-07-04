"""Modal premise test (Week-1 pilot gate (a), proposal §1.7).

Measures the discriminator–generator asymmetry on Qwen2.5-1.5B base:

    p_disc(x)            = P_base(answer | x, y_oracle)
    p_disc_corrupted(x)  = P_base(answer | x, y_corrupted)

For each problem, we feed `prompt = x ⊕ y_oracle` to the base model
and measure how often the model emits the correct boxed answer in N
sampled completions (the *behavioral* test from proposal §2.7.1, not
the calibration-sensitive log-prob test). The corruption variant
alters one critical numeric step in `y_oracle`; the step-shuffle
variant randomizes solution-step order.

Decision rule (proposal §1.7):
    mean(p_disc) ≥ 0.5 AND
    mean(p_disc) - mean(p_disc_corrupted) ≥ 0.2
on at least 60% of problems, paired sign test α = 0.05.

The orchestration logic (running the same test on synthetic + step-
shuffle controls; aggregating; deciding the gate) lives in
``scripts/run_premise_test.py``. This module owns the GPU
forward-pass primitive only.

Cost model (Qwen2.5-1.5B, 4×H100):
- 100 problems × 8 samples × 200 tok × 4 conditions (oracle, corrupted,
  shuffled, no-CoT) ≈ 640k tokens
- vLLM batched ≈ 120k tok/s aggregate → ≈ 5.5 s + overhead
- ≈ $0.10 — negligible against the $30 premise-test budget line.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

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


def corrupt_solution_one_step(solution: str, *, seed: int = 0) -> str:
    """Alter one critical numeric token in `solution` to break a
    causal step. Reproducible given seed.

    The corruption must change *some* number (not the final answer)
    so that the chain becomes wrong but still looks plausible. We do
    this with a regex over base-10 integers.
    """
    import random

    rng = random.Random(seed)
    nums = list(_LATEX_NUM.finditer(solution))
    if not nums:
        return solution
    # Avoid corrupting the very last number — that's most likely the
    # final answer; we want the *intermediate* steps to be wrong.
    candidates = nums[:-1] if len(nums) > 1 else nums
    target = rng.choice(candidates)
    original = int(target.group(1))
    # Shift by a small nonzero delta (1–9), keeping sign.
    delta = rng.choice([1, 2, 3, -1, -2, -3, 5, -5, 7])
    new = max(0, original + delta)
    if new == original:
        new += 1
    span = target.span(1)
    return solution[: span[0]] + str(new) + solution[span[1] :]


def shuffle_solution_steps(solution: str, *, seed: int = 0) -> str:
    """Randomize solution-step ordering (paragraph or numbered-line
    granularity). Used for the step-shuffle control."""
    import random

    rng = random.Random(seed)
    # Try numbered-step granularity first (NuminaMath default).
    parts = re.split(r"(?=^\s*\d+\.\s)", solution, flags=re.MULTILINE)
    parts = [p for p in parts if p.strip()]
    if len(parts) >= 3:
        rng.shuffle(parts)
        return "".join(parts)
    # Fallback: shuffle by sentences.
    sentences = re.split(r"(?<=[.!?])\s+", solution)
    sentences = [s for s in sentences if s.strip()]
    if len(sentences) >= 3:
        rng.shuffle(sentences)
        return " ".join(sentences)
    return solution


if modal is not None:
    app = modal.App(f"{APP_NAME}-premise", image=IMAGE)

    @app.function(
        gpu=gpu_spec("pilot"),
        volumes=VOLUMES,
        secrets=get_secrets(),
        timeout=3600,
        memory=32_000,
    )
    def run_premise_test(
        problems: list[dict],
        *,
        model_id: str = "Qwen/Qwen2.5-1.5B",
        n_samples: int = 8,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.95,
        seed: int = 1337,
        save_to: str = "/cache/premise/premise_test_1p5b.jsonl",
    ) -> dict:
        """Compute per-problem ``p_disc`` under four conditions:

        - ``no_cot``: only the problem statement (= ``p_gen`` upper bound).
        - ``oracle``: x ⊕ y_oracle.
        - ``corrupted``: x ⊕ y_corrupted (one critical step flipped).
        - ``shuffled``: x ⊕ y_step_shuffle (steps reordered).

        Each condition draws ``n_samples`` continuations; we verify each
        via SymPy and report the per-problem fraction of correct
        completions. Per-problem records → JSONL on the cache volume.

        Memorization guard: ``corrupted`` and ``shuffled`` are
        pre-registered controls. If
        ``p_disc(oracle) - p_disc(corrupted) < 0.2``, the base may be
        matching on terminal-answer cues (proposal §1.7).
        """
        from vllm import LLM, SamplingParams

        from reflex_rlvr.verifier import verify

        out_path = Path(save_to)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        llm = LLM(
            model=model_id,
            dtype="bfloat16",
            seed=seed,
            tensor_parallel_size=4,
            max_model_len=4096,
            trust_remote_code=True,
        )
        sampling_params = SamplingParams(
            n=n_samples,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
        )

        # Build the four condition prompt sets.
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

        conditions = ["no_cot", "oracle", "corrupted", "shuffled"]
        all_prompts: list[str] = []
        prompt_meta: list[tuple[int, str]] = []  # (problem_idx, condition)
        for i, p in enumerate(problems):
            no_cot = build_prompt(p["problem"], None)
            oracle = build_prompt(p["problem"], p["solution"])
            corrupted = build_prompt(
                p["problem"],
                corrupt_solution_one_step(p["solution"], seed=seed + i),
            )
            shuffled = build_prompt(
                p["problem"],
                shuffle_solution_steps(p["solution"], seed=seed + i),
            )
            all_prompts.extend([no_cot, oracle, corrupted, shuffled])
            for c in conditions:
                prompt_meta.append((i, c))

        gen_t0 = time.time()
        outputs = llm.generate(all_prompts, sampling_params)
        gen_elapsed = time.time() - gen_t0

        # Aggregate per (problem, condition).
        per_problem: list[dict] = [
            {
                "id": p.get("id"),
                "year": p.get("year"),
                "answer": p["answer"],
                "p_disc_no_cot": None,
                "p_disc_oracle": None,
                "p_disc_corrupted": None,
                "p_disc_shuffled": None,
            }
            for p in problems
        ]
        for output, (i, cond) in zip(outputs, prompt_meta):
            n_correct = 0
            for completion in output.outputs:
                res = verify(
                    {"domain": "math", "answer": problems[i]["answer"]},
                    completion.text,
                )
                n_correct += int(res.accepted)
            p_correct = n_correct / max(1, len(output.outputs))
            per_problem[i][f"p_disc_{cond}"] = p_correct

        with out_path.open("w", encoding="utf-8") as fh:
            for row in per_problem:
                fh.write(json.dumps(row) + "\n")

        try:
            VOLUMES["/cache"].commit()
        except Exception:  # noqa: BLE001
            pass

        # Aggregate stats.
        def _mean(key: str) -> float:
            vals = [r[key] for r in per_problem if r[key] is not None]
            return round(sum(vals) / max(1, len(vals)), 4)

        total_elapsed = time.time() - t0
        return {
            "n_problems": len(problems),
            "model_id": model_id,
            "n_samples_per_condition": n_samples,
            "mean_p_disc_no_cot": _mean("p_disc_no_cot"),
            "mean_p_disc_oracle": _mean("p_disc_oracle"),
            "mean_p_disc_corrupted": _mean("p_disc_corrupted"),
            "mean_p_disc_shuffled": _mean("p_disc_shuffled"),
            "delta_oracle_minus_corrupted": round(
                _mean("p_disc_oracle") - _mean("p_disc_corrupted"), 4
            ),
            "gen_elapsed_s": round(gen_elapsed, 2),
            "total_elapsed_s": round(total_elapsed, 2),
            "estimated_cost_usd": estimate_cost(
                "H100", 4, total_elapsed / 3600.0
            ),
            "save_to": str(out_path),
        }

    @app.function(
        gpu=gpu_spec("pilot"),
        volumes=VOLUMES,
        secrets=get_secrets(),
        timeout=3600,
        memory=32_000,
    )
    def run_logprob_premise_test(
        problems: list[dict],
        *,
        model_id: str = "Qwen/Qwen2.5-1.5B",
        seed: int = 1337,
        save_to: str = "/cache/premise/logprob_premise_1p5b.jsonl",
    ) -> dict:
        """Compute per-problem mean log-prob of the answer under four conditions.

        Conditions:
        - ``no_cot``: only the problem statement.
        - ``oracle``: x ⊕ y_oracle.
        - ``corrupted``: x ⊕ y_corrupted (one critical step flipped).
        - ``shuffled``: x ⊕ y_step_shuffle (steps reordered).

        For each condition, we score the *answer string* (``\\boxed{N}``) via
        prompt log-probs (max_tokens=1 so the model does no generation).
        Per-problem records → JSONL on the cache volume.

        Gate: delta > 0 AND pct_oracle_beats_corrupted > 0.6 AND
        Wilcoxon one-sided p < 0.05.
        """
        from scipy.stats import wilcoxon  # noqa: PLC0415 — scipy in worker image only
        from vllm import LLM, SamplingParams

        out_path = Path(save_to)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        llm = LLM(
            model=model_id,
            dtype="bfloat16",
            seed=seed,
            tensor_parallel_size=4,
            max_model_len=4096,
            trust_remote_code=True,
        )
        tokenizer = llm.get_tokenizer()

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

        # Use prompt_logprobs=None to get logprobs for ALL prompt tokens
        # (including the actual input token at each position), not just the
        # top-1. With prompt_logprobs=1 vLLM returns only the argmax's
        # logprob — useless for scoring a *specific* answer string whose
        # tokens may not be the model's argmax.
        scoring_params = SamplingParams(
            prompt_logprobs=0,  # 0 => return logprob of the actual token only
            max_tokens=1,
            temperature=0.0,
            seed=seed,
        )

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

                # Robust boundary: tokenize both prompt and full_text the
                # same way (with special tokens) and use the difference in
                # length as the count of answer tokens. This sidesteps any
                # BOS / chat-template asymmetry between
                # `add_special_tokens=True` and `False`.
                full_ids = tokenizer.encode(full_text)
                prompt_ids = tokenizer.encode(prompt)
                n_ans = len(full_ids) - len(prompt_ids)
                # Sanity: n_ans should be small and positive.
                if n_ans <= 0:
                    n_ans = max(1, len(tokenizer.encode(answer_string,
                                                        add_special_tokens=False)))
                # The answer-token positions in prompt_logprobs are the
                # LAST n_ans positions (regardless of any BOS offset).
                answer_lp_list: list[float] = []
                prompt_lp = result.prompt_logprobs  # type: ignore[attr-defined]
                if prompt_lp is not None and len(prompt_lp) >= n_ans:
                    answer_positions = prompt_lp[-n_ans:]
                    for pos_idx, pos_lp in enumerate(answer_positions):
                        if pos_lp is None:
                            continue
                        # pos_lp is a dict {token_id: Logprob}. With
                        # prompt_logprobs=0 vLLM returns only the actual
                        # input token's logprob, so any value() is correct.
                        # We still look up by the actual token id where
                        # possible to be defensive against vLLM returning
                        # extra entries.
                        actual_id = full_ids[len(full_ids) - n_ans + pos_idx]
                        lp_obj = pos_lp.get(actual_id)
                        if lp_obj is None:
                            # Fallback: take any value (should match the
                            # actual token under prompt_logprobs=0).
                            lp_obj = next(iter(pos_lp.values()))
                        # vLLM Logprob has a .logprob attribute; some
                        # versions also accept a raw float, handle both.
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
                fh.write(json.dumps(row) + "\n")

        try:
            VOLUMES["/cache"].commit()
        except Exception:  # noqa: BLE001
            pass

        # Aggregate stats.
        def _mean(key: str) -> float:
            vals = [r[key] for r in per_problem if r.get(key) is not None]
            return round(sum(vals) / max(1, len(vals)), 4)

        deltas = [r["delta_oracle_corrupted"] for r in per_problem]
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
            "n_problems": len(problems),
            "model_id": model_id,
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
            "estimated_cost_usd": estimate_cost(
                "H100", 4, total_elapsed / 3600.0
            ),
            "save_to": str(out_path),
        }

    @app.local_entrypoint()
    def main_logprob(
        problems_jsonl: str,
        *,
        model_id: str = "Qwen/Qwen2.5-1.5B",
        save_to: str = "/cache/premise/logprob_premise_1p5b.jsonl",
    ) -> None:
        path = Path(problems_jsonl)
        if not path.exists():
            raise FileNotFoundError(
                f"problems file not found: {path}. Run `python "
                "scripts/fetch_data.py` first."
            )
        problems = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        # Logprob premise test only runs on problems with a CoT solution.
        problems = [p for p in problems if p.get("solution")]
        print(
            f"[logprob_premise_test] {len(problems)} problems → 4 conditions "
            f"(prompt log-prob scoring) on {model_id}"
        )
        summary = run_logprob_premise_test.remote(
            problems,
            model_id=model_id,
            save_to=save_to,
        )
        print("[logprob_premise_test] summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        gate = summary["gate_logprob_passed"]
        delta = summary["delta_oracle_minus_corrupted"]
        pct = summary["pct_oracle_beats_corrupted"]
        pval = summary["wilcoxon_pvalue"]
        print(
            f"\n[gate (logprob)] Δ(oracle−corrupted) = {delta:.4f}; "
            f"pct_oracle_beats = {pct:.3f}; Wilcoxon p = {pval:.4f}; "
            f"PASSED={gate}"
        )
        if not gate:
            print(
                "[gate (logprob)] FAILED — log-prob discriminator–generator "
                "asymmetry premise rejected."
            )

    @app.local_entrypoint()
    def main(
        problems_jsonl: str,
        *,
        model_id: str = "Qwen/Qwen2.5-1.5B",
        n_samples: int = 8,
        save_to: str = "/cache/premise/premise_test_1p5b.jsonl",
    ) -> None:
        path = Path(problems_jsonl)
        if not path.exists():
            raise FileNotFoundError(
                f"problems file not found: {path}. Run `python "
                "scripts/fetch_data.py` first."
            )
        problems = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        # Premise test only runs on problems with a CoT solution.
        problems = [p for p in problems if p.get("solution")]
        print(
            f"[premise_test] {len(problems)} problems → 4 conditions × "
            f"{n_samples} samples on {model_id}"
        )
        summary = run_premise_test.remote(
            problems,
            model_id=model_id,
            n_samples=n_samples,
            save_to=save_to,
        )
        print("[premise_test] summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        # Print the gate decision per proposal §1.7.
        d = summary["delta_oracle_minus_corrupted"]
        oracle_mean = summary["mean_p_disc_oracle"]
        gate_a_passed = oracle_mean >= 0.5 and d >= 0.2
        print(
            f"\n[gate (a)] mean p_disc(oracle) = {oracle_mean:.3f}; "
            f"Δ(oracle - corrupted) = {d:.3f}; "
            f"PASSED={gate_a_passed}"
        )
        if not gate_a_passed:
            print(
                "[gate (a)] FAILED — discriminator–generator asymmetry "
                "premise rejected. Pivot per proposal §10."
            )

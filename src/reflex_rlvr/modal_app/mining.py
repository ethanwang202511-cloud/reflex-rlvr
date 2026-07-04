"""Modal mining: vLLM batched ``pass@k`` sweep on a candidate problem
set.

This is the *first real GPU job* — used for both:
- Hard-set mining (`base_pass@1024 == 0` filter on candidate problems).
- Pre-pilot crossover gate (`pass@8` measurement on the trained 1.5B).

The function is parameterized by `k` (samples per problem), `max_tokens`,
and the model id; pilot defaults match `configs/pilot.yaml`.

Cost model (Qwen2.5-1.5B, 200 tok/sample, vLLM batched on 4×H100):
- 100 problems × 1024 samples × 200 tok ≈ 20M tokens
- vLLM batched ≈ 30k tok/s/H100 × 4 GPUs ≈ 120k tok/s aggregate
- ≈ 167 s = 2.8 min wall-clock + cold-start overhead
- ≈ $0.40 H100·hr ≈ $4 — well under the $40 mining-line budget for
  the pilot.

For the 7B main-run hard-set mining (50k problems × 1024 samples on
8×H100), the budget is $2,680 (architecture §8.1).
"""

from __future__ import annotations

import json
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


if modal is not None:
    app = modal.App(f"{APP_NAME}-mining", image=IMAGE)

    @app.function(
        gpu=gpu_spec("pilot"),
        volumes=VOLUMES,
        secrets=get_secrets(),
        timeout=3600,
        memory=32_000,
    )
    def mine_pass_at_k(
        problems: list[dict],
        *,
        model_id: str = "Qwen/Qwen2.5-1.5B",
        k: int = 1024,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.95,
        seed: int = 1337,
        save_to: str = "/cache/mining/h_k_pilot_pass1024.jsonl",
    ) -> dict:
        """Run pass@k sweep on the supplied problems.

        Each ``problem`` is ``{id, problem, answer, ...}``. We sample
        ``k`` continuations per problem at the configured temperature,
        verify each via the SymPy verifier, and write per-problem
        per-sample correctness records to ``save_to``.

        Returns a summary dict with `n_problems`, `wall_clock_s`,
        `n_problems_solved_at_least_once`, etc.
        """
        from vllm import LLM, SamplingParams  # heavy import inside Modal worker

        from reflex_rlvr.verifier import verify  # works in the image

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

        prompts = [
            (
                "Solve the following problem. Show your reasoning briefly, "
                "then put the final integer answer in \\boxed{}.\n\n"
                f"Problem: {p['problem']}\n\nSolution: "
            )
            for p in problems
        ]

        # Chunked generation: a single generate() with n=k creates
        # n_problems × k simultaneous sequences which stalls the vLLM
        # scheduler at high k (e.g. 69 × 1024 = 70k). Chunk into
        # batches of CHUNK samples per prompt; one generate() call per
        # chunk. This keeps the live-sequence count reasonable.
        CHUNK = 16
        gen_t0 = time.time()
        all_completions: list[list] = [[] for _ in problems]
        n_chunks = (k + CHUNK - 1) // CHUNK
        for chunk_idx in range(n_chunks):
            chunk_n = min(CHUNK, k - chunk_idx * CHUNK)
            sp = SamplingParams(
                n=chunk_n,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=seed + chunk_idx,
            )
            chunk_outputs = llm.generate(prompts, sp)
            for j, out in enumerate(chunk_outputs):
                all_completions[j].extend(out.outputs)
        gen_elapsed = time.time() - gen_t0

        verify_t0 = time.time()
        results: list[dict] = []
        n_solved_any = 0
        with out_path.open("w", encoding="utf-8") as fh:
            for problem, completions in zip(problems, all_completions):
                samples_correct: list[int] = []
                for completion in completions:
                    text = completion.text
                    res = verify(
                        {"domain": "math", "answer": problem["answer"]},
                        text,
                    )
                    samples_correct.append(int(res.accepted))
                solved_any = any(samples_correct)
                n_solved_any += int(solved_any)
                row = {
                    "id": problem.get("id"),
                    "year": problem.get("year"),
                    "problem_number": problem.get("problem_number"),
                    "answer": problem["answer"],
                    "n_samples": len(samples_correct),
                    "n_correct": int(sum(samples_correct)),
                    "samples_correct": samples_correct,
                }
                fh.write(json.dumps(row) + "\n")
                results.append(row)
        verify_elapsed = time.time() - verify_t0
        total_elapsed = time.time() - t0

        # Persist the cache volume so the writes survive after function exits.
        try:
            VOLUMES["/cache"].commit()
        except Exception:  # noqa: BLE001
            pass

        return {
            "n_problems": len(problems),
            "k": k,
            "model_id": model_id,
            "n_solved_any": n_solved_any,
            "n_solved_zero": len(problems) - n_solved_any,
            "gen_elapsed_s": round(gen_elapsed, 2),
            "verify_elapsed_s": round(verify_elapsed, 2),
            "total_elapsed_s": round(total_elapsed, 2),
            "estimated_cost_usd": estimate_cost(
                "H100", 4, total_elapsed / 3600.0
            ),
            "save_to": str(out_path),
        }

    @app.local_entrypoint()
    def main(
        problems_jsonl: str,
        *,
        model_id: str = "Qwen/Qwen2.5-1.5B",
        k: int = 1024,
        max_tokens: int = 200,
        temperature: float = 0.8,
        top_p: float = 0.95,
        seed: int = 1337,
        save_to: str = "/cache/mining/h_k_pilot_pass1024.jsonl",
    ) -> None:
        """Local entrypoint — invoked by ``modal run scripts/run_mining.py``.

        Reads problems from a JSONL file on the *local* machine,
        forwards to the GPU worker, prints summary on return.
        """
        path = Path(problems_jsonl)
        if not path.exists():
            raise FileNotFoundError(
                f"problems file not found: {path}. Run "
                "`python scripts/fetch_data.py` first."
            )
        problems = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        print(
            f"[mine_pass_at_k] {len(problems)} problems → pass@{k} on "
            f"{model_id} ({gpu_spec('pilot') or 'CPU'})"
        )
        summary = mine_pass_at_k.remote(
            problems,
            model_id=model_id,
            k=k,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            save_to=save_to,
        )
        print("[mine_pass_at_k] summary:")
        for k_, v in summary.items():
            print(f"  {k_}: {v}")

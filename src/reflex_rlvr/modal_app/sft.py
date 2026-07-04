"""SFT pipeline for the verdict-recovery experiment (Round 8, Phase 2).

Goal. Test the working hypothesis from §5.3 of the paper:

  "Qwen pretraining and R1-style distillation install a token-level routing
   from chain-perception to a YES/NO verdict head."

If a small SFT pass with graded-judgment data on a non-Qwen base recovers
verdict capability, the hypothesis is confirmed. Round 6 measured
Llama-3.1-8B base at Δ_verdict ≈ 0 (no verdict signal); this experiment
asks whether a 1-epoch LoRA SFT on ~1000 (problem, candidate-chain,
YES/NO) examples is enough to install the routing.

Design choice (per FUTURE_EXPERIMENTS / Round 8 plan, option 2b).

Training data is constructed from NuminaMath AMC/AIME problems
(``data/aime/numina_amc_aime.jsonl``, 4070 problems, fully disjoint
from the 51-problem eval set ``data/aime/h_k_pilot_filtered.jsonl``).
For each sampled problem we emit one (problem, oracle_solution,
YES) and one (problem, consistent_corruption(oracle), NO) row, using
the same ``corrupt_solution_consistent`` helper as the Probe-2
verdict test. The verdict prompt format mirrors ``_VERDICT_TEMPLATE``
in ``discrimination_v2.run_verdict_test`` exactly, so the SFT objective
matches the probe being evaluated.

LoRA hyperparameters per Round-8 plan: rank 16, alpha 32, target
attention and MLP modules, lr 1e-4, 1 epoch, batch size 4, AdamW.

The merged adapter is saved to a Modal volume at
``/checkpoints/llama_sft_verdict_lora/`` and pulled by the
post-SFT verdict probe.
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


_VERDICT_TEMPLATE = (
    "You are a careful math grader. Read the problem and the proposed "
    "solution, then judge whether the proposed solution is correct.\n\n"
    "Problem: {problem}\n\n"
    "Proposed solution: {solution}\n\n"
    "Is the proposed solution correct? Answer YES or NO.\n\n"
    "Answer:"
)


def _extract_answer_from_solution(solution: str):
    """Pull the integer in the last ``\\boxed{...}`` of a NuminaMath
    solution. Returns None if no boxed integer present."""
    import re
    matches = list(re.finditer(r"\\boxed\{([^}]*)\}", solution))
    if not matches:
        return None
    raw = matches[-1].group(1).strip()
    # Accept plain ints; reject fractions, expressions, etc.
    try:
        return int(raw)
    except (ValueError, TypeError):
        return None


def build_sft_dataset(
    numina_jsonl: str,
    out_jsonl: str,
    *,
    n_problems: int = 500,
    seed: int = 1337,
) -> dict:
    """Construct (problem, candidate_chain, YES/NO) triples for SFT.

    Two rows per sampled problem: one positive (oracle, YES) and one
    negative (consistent_corruption, NO). Skips problems whose
    solutions don't terminate in ``\\boxed{<int>}``.
    """
    import random
    from reflex_rlvr.modal_app.discrimination_v2 import (
        corrupt_solution_consistent,
    )

    rng = random.Random(seed)
    rows_in = [
        json.loads(l)
        for l in Path(numina_jsonl).read_text().splitlines()
        if l.strip()
    ]
    rng.shuffle(rows_in)

    examples: list[dict] = []
    used = 0
    for r in rows_in:
        if used >= n_problems:
            break
        sol = r.get("solution", "")
        ans = _extract_answer_from_solution(sol)
        if ans is None or ans <= 0:
            continue
        # Skip very long solutions to keep prompts under 2048 tokens.
        if len(sol) > 3000:
            continue
        prob = r.get("problem", "").strip()
        if not prob:
            continue
        corrupted, _ = corrupt_solution_consistent(
            sol, ans, seed=seed + used,
        )
        examples.append({
            "problem": prob,
            "solution": sol,
            "label": "YES",
        })
        examples.append({
            "problem": prob,
            "solution": corrupted,
            "label": "NO",
        })
        used += 1

    out_path = Path(out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for ex in examples:
            fh.write(json.dumps(ex) + "\n")
    return {
        "n_problems": used,
        "n_examples": len(examples),
        "out_jsonl": str(out_path),
    }


if modal is not None:
    app = modal.App(f"{APP_NAME}-sft-verdict", image=IMAGE)

    @app.function(
        gpu=gpu_spec("pilot"),
        volumes=VOLUMES,
        secrets=get_secrets(),
        timeout=4 * 3600,
        memory=64_000,
    )
    def run_sft_verdict(
        train_examples: list[dict],
        *,
        model_id: str = "meta-llama/Llama-3.1-8B",
        seed: int = 1337,
        n_epochs: int = 1,
        lr: float = 1e-4,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        batch_size: int = 4,
        grad_accum_steps: int = 4,
        save_to: str = "/checkpoints/llama_sft_verdict_lora",
    ) -> dict:
        """SFT a LoRA adapter on (problem, solution, YES/NO) data.

        Saves the adapter (NOT a merged model) so the eval can either
        merge at runtime or load with PEFT.
        """
        import os
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, TaskType
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling,
        )

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        t0 = time.time()

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Build (prompt + " " + label) → mask prompt tokens during loss.
        # We want the loss to only fire on the single-token label.
        def encode(example):
            prompt = _VERDICT_TEMPLATE.format(
                problem=example["problem"],
                solution=example["solution"],
            )
            target = " " + example["label"]  # leading space matches verdict probe
            prompt_ids = tokenizer.encode(
                prompt, add_special_tokens=True,
            )
            target_ids = tokenizer.encode(
                target, add_special_tokens=False,
            )
            input_ids = prompt_ids + target_ids
            # Truncate prompt from the left if total > 2048.
            max_len = 2048
            if len(input_ids) > max_len:
                overflow = len(input_ids) - max_len
                prompt_ids = prompt_ids[overflow:]
                input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + list(target_ids)
            # Sanity: lengths match.
            assert len(input_ids) == len(labels)
            return {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": [1] * len(input_ids),
            }

        ds = Dataset.from_list(train_examples).map(
            encode,
            remove_columns=["problem", "solution", "label"],
        )

        # Pad-to-max-in-batch collator.
        def collate(batch):
            max_len = max(len(b["input_ids"]) for b in batch)
            input_ids, labels, attn = [], [], []
            pad_id = tokenizer.pad_token_id
            for b in batch:
                n_pad = max_len - len(b["input_ids"])
                input_ids.append(b["input_ids"] + [pad_id] * n_pad)
                labels.append(b["labels"] + [-100] * n_pad)
                attn.append(b["attention_mask"] + [0] * n_pad)
            return {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
            }

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        model.gradient_checkpointing_enable()

        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        out_path = Path(save_to)
        out_path.mkdir(parents=True, exist_ok=True)

        args = TrainingArguments(
            output_dir=str(out_path),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum_steps,
            num_train_epochs=n_epochs,
            learning_rate=lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            logging_steps=10,
            save_strategy="no",
            bf16=True,
            seed=seed,
            report_to=[],
            remove_unused_columns=False,
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=ds,
            data_collator=collate,
        )
        trainer.train()

        # Save adapter.
        model.save_pretrained(str(out_path))
        tokenizer.save_pretrained(str(out_path))
        try:
            VOLUMES["/checkpoints"].commit()
        except Exception:  # noqa: BLE001
            pass

        elapsed = time.time() - t0
        return {
            "model_id": model_id,
            "n_train": len(train_examples),
            "n_epochs": n_epochs,
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lr": lr,
            "elapsed_s": round(elapsed, 1),
            "estimated_cost_usd": estimate_cost("H100", 4, elapsed / 3600.0),
            "save_to": str(out_path),
        }

    @app.function(
        gpu=gpu_spec("pilot"),
        volumes=VOLUMES,
        secrets=get_secrets(),
        timeout=3600,
        memory=64_000,
    )
    def run_verdict_test_with_adapter(
        problems: list[dict],
        *,
        model_id: str = "meta-llama/Llama-3.1-8B",
        adapter_path: str = "/checkpoints/llama_sft_verdict_lora",
        seed: int = 1337,
        save_to: str = "/cache/premise/verdict_llama_3p1_8b_sft.jsonl",
    ) -> dict:
        """Run the verdict probe on a base model + LoRA adapter.

        We can't easily pass a PEFT-merged model to vLLM, so we merge
        the adapter into the base and re-save the merged weights to a
        local path, then point vLLM at it. The merged model is
        discarded at end-of-function.
        """
        import math
        import os
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from vllm import LLM, SamplingParams
        from scipy.stats import wilcoxon

        from reflex_rlvr.modal_app.discrimination_v2 import (
            corrupt_solution_consistent,
            shuffle_solution_steps,
        )

        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        t0 = time.time()

        # Merge.
        print(f"[verdict_sft] Loading base {model_id} and adapter {adapter_path}")
        base = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
        merged = PeftModel.from_pretrained(base, adapter_path)
        merged = merged.merge_and_unload()
        merged_dir = "/tmp/llama_sft_merged"
        merged.save_pretrained(merged_dir, safe_serialization=True)
        tokenizer_hf = AutoTokenizer.from_pretrained(adapter_path)
        tokenizer_hf.save_pretrained(merged_dir)
        del base, merged
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # Re-instantiate vLLM on the merged checkpoint.
        llm = LLM(
            model=merged_dir,
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

        def score_verdict_token(prompt: str, token_str: str) -> float:
            full_text = prompt + token_str
            result = llm.generate([full_text], scoring_params)[0]
            full_ids = tokenizer.encode(full_text)
            prompt_ids = tokenizer.encode(prompt)
            n_ans = len(full_ids) - len(prompt_ids)
            if n_ans <= 0:
                n_ans = max(
                    1, len(tokenizer.encode(token_str, add_special_tokens=False))
                )
            answer_lp_list: list[float] = []
            prompt_lp = result.prompt_logprobs
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
                if answer_lp_list else float("nan")
            )

        def verdict_logit(prompt: str) -> float:
            logp_yes = score_verdict_token(prompt, " YES")
            logp_no = score_verdict_token(prompt, " NO")
            return round(logp_yes - logp_no, 4)

        per_problem: list[dict] = []
        for i, problem in enumerate(problems):
            n_oracle = int(problem["answer"])
            corrupted_solution, _ = corrupt_solution_consistent(
                problem["solution"], n_oracle, seed=seed + i,
            )
            shuffled_solution = shuffle_solution_steps(
                problem["solution"], seed=seed + i,
            )
            prompt_oracle = _VERDICT_TEMPLATE.format(
                problem=problem["problem"], solution=problem["solution"],
            )
            prompt_corrupted = _VERDICT_TEMPLATE.format(
                problem=problem["problem"], solution=corrupted_solution,
            )
            prompt_shuffled = _VERDICT_TEMPLATE.format(
                problem=problem["problem"], solution=shuffled_solution,
            )
            logit_oracle = verdict_logit(prompt_oracle)
            logit_corrupted = verdict_logit(prompt_corrupted)
            logit_shuffled = verdict_logit(prompt_shuffled)
            per_problem.append({
                "id": problem.get("id"),
                "answer": n_oracle,
                "logit_oracle": logit_oracle,
                "logit_corrupted": logit_corrupted,
                "logit_shuffled": logit_shuffled,
                "delta_oracle_minus_corrupted": round(
                    logit_oracle - logit_corrupted, 4
                ),
            })

        out_path = Path(save_to)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as fh:
            for r in per_problem:
                clean = {
                    k: (None if isinstance(v, float) and math.isnan(v) else v)
                    for k, v in r.items()
                }
                fh.write(json.dumps(clean, allow_nan=False) + "\n")
        try:
            VOLUMES["/cache"].commit()
        except Exception:  # noqa: BLE001
            pass

        deltas = [
            r["delta_oracle_minus_corrupted"]
            for r in per_problem
            if r["delta_oracle_minus_corrupted"] is not None
            and not (isinstance(r["delta_oracle_minus_corrupted"], float)
                     and math.isnan(r["delta_oracle_minus_corrupted"]))
        ]
        mean_delta = round(sum(deltas) / max(1, len(deltas)), 4) if deltas else float("nan")
        pct = round(
            sum(1 for r in per_problem if r["logit_oracle"] > r["logit_corrupted"])
            / max(1, len(per_problem)), 4,
        )
        try:
            wp = float(wilcoxon(deltas, alternative="greater").pvalue)
        except Exception:  # noqa: BLE001
            wp = float("nan")

        elapsed = time.time() - t0
        return {
            "model_id": model_id,
            "adapter_path": adapter_path,
            "n_problems": len(problems),
            "mean_delta_verdict": mean_delta,
            "pct_oracle_above_corrupted": pct,
            "wilcoxon_pvalue": round(wp, 6),
            "elapsed_s": round(elapsed, 1),
            "estimated_cost_usd": estimate_cost("H100", 4, elapsed / 3600.0),
            "save_to": str(out_path),
        }

    @app.local_entrypoint()
    def main_sft(
        train_jsonl: str,
        *,
        model_id: str = "meta-llama/Llama-3.1-8B",
        seed: int = 1337,
        n_epochs: int = 1,
        lr: float = 1e-4,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        save_to: str = "/checkpoints/llama_sft_verdict_lora",
    ) -> None:
        examples = [
            json.loads(l)
            for l in Path(train_jsonl).read_text().splitlines()
            if l.strip()
        ]
        print(f"[sft] training on {len(examples)} examples → {save_to}")
        summary = run_sft_verdict.remote(
            examples,
            model_id=model_id,
            seed=seed,
            n_epochs=n_epochs,
            lr=lr,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            save_to=save_to,
        )
        for k, v in summary.items():
            print(f"  {k}: {v}")

    @app.local_entrypoint()
    def main_verdict_sft(
        problems_jsonl: str,
        *,
        model_id: str = "meta-llama/Llama-3.1-8B",
        adapter_path: str = "/checkpoints/llama_sft_verdict_lora",
        seed: int = 1337,
        save_to: str = "/cache/premise/verdict_llama_3p1_8b_sft.jsonl",
    ) -> None:
        problems = [
            json.loads(l)
            for l in Path(problems_jsonl).read_text().splitlines()
            if l.strip()
        ]
        problems = [p for p in problems if p.get("solution")]
        print(f"[verdict_sft] {len(problems)} eval problems on {model_id}+{adapter_path}")
        summary = run_verdict_test_with_adapter.remote(
            problems,
            model_id=model_id,
            adapter_path=adapter_path,
            seed=seed,
            save_to=save_to,
        )
        for k, v in summary.items():
            print(f"  {k}: {v}")

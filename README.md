# Re-measuring the Discrimination Ceiling: Can Base LLMs Verify Their Own Math?

**Accepted at [AI4Math Workshop](https://ai4mathworkshopicml2026.github.io/) and [RLxF Workshop](https://rlxf-workshop.github.io/), ICML 2026**

Ethan Y. Wang and Aayan Alwani

Corresponding author: ethanwang202511@gmail.com

## Abstract

We test whether base LLMs can verify their own math reasoning chains. A pre-registered behavioral test on Qwen2.5-1.5B base over AIME hard problems fails on every sub-criterion. But four corrective probes across 11 model configurations (Qwen2.5, DeepSeek-R1-Distill, Llama-3.1, Mistral, Gemma-2; 1.5B--72B) show the chain-content signal is universally present -- the original "discrimination ceiling" was a measurement-gating phenomenon. Qwen and R1-Distill models can verdict correctness; non-Qwen base models cannot despite perceiving chain quality. A 1K-example LoRA SFT recovers the verdict route.

**TL;DR:** Base LLMs perceive math-chain quality universally but verbalize YES/NO only with Qwen/R1 pretraining; a 1K-example LoRA recovers the route.

## Repository structure

```
src/                   Source package (reflex_rlvr)
  eval/                Pass@k evaluation
  gsi/                 Gradient-spectral initialization
  latent/              Latent register diagnostics, halt head, cosine annealing
  mining/              Hard-set mining utilities
  modal_app/           Modal GPU entrypoints (mining, premise test, SFT, verdict)
  translator/          Problem format translation
  verifier/            Sympy, code, and Lean verifiers with router
scripts/               Experiment launcher scripts
configs/               YAML experiment configurations
results/               Raw results (JSONL, CSV, JSON, Markdown summaries)
figures/               Figure rendering scripts
tests/                 Unit tests
data/                  Data directory (gitignored; see data/README.md)
```

## Installation

```bash
# With uv (recommended)
uv sync

# Or with pip
pip install -e ".[dev]"
```

## Running tests

```bash
pytest
```

## Reproducing key results

Experiment scripts live in `scripts/`. Each script is self-contained and documents its Modal GPU requirements.

| Experiment | Script | Description |
|---|---|---|
| Hard-set mining (pass@1024) | `scripts/run_mining.py` | Identify AIME problems unsolvable by Qwen2.5-1.5B base |
| Behavioral premise test | `scripts/run_premise_test.py` | Pre-registered gate (a) discrimination test |
| Logprob premise test | `scripts/run_logprob_premise.py` | Logprob-based premise test across 11 configs |
| Consistent corruption | `scripts/run_consistent_corruption.py` | Corrective probe: answer-preserving corruption |
| Per-token surprise | `scripts/run_per_token_surprise.py` | Corrective probe: per-token surprise analysis |
| Format ablation | `scripts/run_format_ablation.py` | Corrective probe: prompt format sensitivity |
| Random-answer corruption | `scripts/run_random_answer_corruption.py` | Corrective probe: random answer corruption |
| Verdict test | `scripts/run_verdict_test.py` | YES/NO verdict extraction across models |
| SFT verdict recovery | `scripts/run_sft_verdict.py` | 1K-example LoRA SFT verdict recovery |
| Gate (a) decision | `scripts/decide_gate_a.py` | Aggregate gate (a) pass/fail decision |
| Seed dispersion | `scripts/analyze_seed_dispersion.py` | Multi-seed robustness analysis |

## Citation

If you use this code or results, please cite:

```bibtex
@inproceedings{wang2026discrimination-ai4math,
  title={Can A Base LLM Verify Its Own Math? A Pre-Registered Discriminator-Generator Asymmetry Test On AIME And Four Corrective Probes},
  author={Wang, Ethan Y. and Alwani, Aayan},
  booktitle={AI4Math Workshop at the International Conference on Machine Learning (ICML)},
  year={2026}
}

@inproceedings{wang2026discrimination-rlxf,
  title={When The Verifier Is The Only Trustworthy Feedback Source: A Self-Teacher RLVR Pilot, A Confounded Logprob Extension, And Four Corrective Probes},
  author={Wang, Ethan Y. and Alwani, Aayan},
  booktitle={RLxF Workshop at the International Conference on Machine Learning (ICML)},
  year={2026}
}
```

## License

Apache-2.0

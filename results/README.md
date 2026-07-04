# results/

Tabular results only. **No figures here** — figures are produced by
scripts in `../figures/` from the data files in this folder.

## Conventions

- **Format.** Prefer CSV for human-readable tables; NPZ / Parquet for
  large dense arrays; JSON for sparse / nested structures (e.g.,
  per-problem rollout records). Whatever format, every file is paired
  with a sibling `.md` that documents columns, units, seed, and
  generating script.
- **One folder per experiment.** Sub-folder names match the
  experiment in `paper/paper.md` and `overview/key_figures.md`.
- **Provenance.** Every result file's sibling `.md` records: the
  generating script, the model checkpoint hash, the dataset version,
  the seed, the Modal app commit, and the date.
- **Reproducibility.** A `make` target or `scripts/<exp>.py --rerun`
  command is preferred over a one-shot run; tabular outputs should be
  rebuildable from the checkpoint and the code without manual
  intervention.

## Layout (planned; populated as experiments run)

```
results/
├── README.md                       ← this file
├── pilot/
│   ├── premise_test_1p5b.csv       ← gate (a): p_disc, p_disc_corrupted, etc.
│   ├── premise_test_1p5b.md        ← provenance / column doc
│   ├── crossover_gate_1p5b.csv     ← gate (b): pass@8 ratios across S_max sweep
│   └── crossover_gate_1p5b.md
├── main_7b/
│   ├── pass_at_k_700_hard_set.csv  ← Figure 1 source
│   ├── thought_trace_smoking_gun.npz
│   ├── cycle_monotonicity.csv      ← Figure 3 source
│   ├── tts_survival.csv            ← Figure 11 source
│   ├── tts_symmetric_diff.csv      ← Figure 12 source
│   └── excluded_problems.csv       ← pre-published exclusion list
├── ablations/
│   └── loo_summary.csv             ← Figure 10 source
├── mechanistic/
│   ├── primitive_stratification.csv
│   ├── sae_feature_trace.npz
│   └── attribution_agreement.csv
├── diagnostics/
│   ├── latent_entropy.csv          ← Figure 7 source
│   ├── halting_entropy.csv         ← Figure 8 source
│   └── forgetting_suite.csv        ← Figure 9 source
├── cross_base/
│   └── llama_3p1_8b_pass_at_k.csv
├── cache/
│   └── (intermediate artifacts; gitignored)
└── checkpoints/
    └── (model checkpoints; gitignored, tracked on HF Hub)
```

## What's here as of 2026-05-03

Nothing yet. No GPU jobs have been launched. Phase-0 local infrastructure
build is in progress; the first results to land will be the pilot
gates.

## Gitignore policy

`cache/` and `checkpoints/` are gitignored. Everything else is
versioned — small CSVs and provenance markdown are part of the audit
trail.

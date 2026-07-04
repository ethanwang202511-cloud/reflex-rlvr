# figures/

Figure scripts and rendered figures. **No figures generated yet** as
of 2026-05-03 — this folder will be populated once `results/` has
tabular data.

## Conventions

- **One script per figure.** `fig<N>_<slug>.py` reads from
  `../results/...` and writes a PDF + PNG to `output/`. The script is
  the source of truth; the rendered file is regenerable.
- **Plotting utilities shared.** `plotting_utils.py` defines colors,
  font sizes, and the canonical NeurIPS template.
- **Figure-to-claim map.** `../overview/key_figures.md` lists each
  planned figure with its supporting claim and data source — keep
  that file in sync when figures are added or renamed.
- **No data lives here.** Every input is read from `../results/`; this
  folder owns rendering only.

## Planned scripts (placeholders, to be created when data lands)

```
figures/
├── README.md                       ← this file
├── plotting_utils.py               ← (TBD) shared style + color palette
├── fig1_pass_at_k_crossover.py     ← Figure 1 — pass@k vs k, all baselines
├── fig2_thought_trace.py           ← Figure 2 — smoking-gun centerpiece (4–5 rows)
├── fig3_cycle_monotonicity.py      ← Figure 3 — Δ pass@K per cycle
├── fig4_primitive_stratification.py← Figure 4 — Stratum-A vs Stratum-B
├── fig5_sae_feature_trace.py       ← Figure 5 — novel-feature firings
├── fig6_attribution_agreement.py   ← Figure 6 — IG vs Thought-Anchors
├── fig7_latent_entropy.py          ← Figure 7 — latent first-step entropy
├── fig8_halting_entropy.py         ← Figure 8 — halting entropy across cycles
├── fig9_forgetting_suite.py        ← Figure 9 — forgetting trajectory
├── fig10_loo_ablation.py           ← Figure 10 — leave-one-out bar chart
├── fig11_tts_survival.py           ← Figure 11 — TTS Kaplan-Meier
├── fig12_tts_symmetric_diff.py     ← Figure 12 — TTS scatter
└── output/                         ← rendered PDFs / PNGs (gitignored)
```

## Build conventions (when implemented)

```bash
# Render every figure
python -m figures.render_all

# Render one figure
python figures/fig1_pass_at_k_crossover.py
```

Figure scripts will adopt the parent Research/ build pipeline
(`make_pdfs_workshops.py` / `make_pdfs_conference.py`) for the
camera-ready PDF embed pass.

## Status

Not started. Will be populated once the Week-1 pilot lands.

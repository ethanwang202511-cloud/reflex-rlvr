"""Build the full reflex-rlvr paper (paper/paper.md) to paper/paper.pdf
using the shared single-column ReportLab pipeline in ../_tools.

No page limit; venue-neutral. Figures are symlinked into a cache dir
under the fig{N}_*.png naming the pipeline expects.
"""
import sys
from pathlib import Path

ROOT = Path("/Users/ethanywang0811/Desktop/Research")
sys.path.insert(0, str(ROOT / "_tools"))
from make_pdfs_conference import _register_fonts, build_paper  # noqa: E402

PROJ = ROOT / "reflex-rlvr"
FIG_SRC = PROJ / "figures" / "output"
FIG_CACHE = PROJ / "paper" / "_figcache"
FIG_CACHE.mkdir(parents=True, exist_ok=True)

# Map figure number -> source PNG filename in figures/output/.
MAPPING = {
    1: "fig1_failure_matrix.png",       # Forest plot: answer-relabeled + verdict panels
    2: "fig2_premise_summary.png",      # 4-condition bar charts across all settings
    3: "fig3_logprob_scatter.png",      # Per-problem logp(oracle) vs logp(corrupted) scatter
    4: "fig4_logprob_distribution.png", # Per-condition box+strip plots
    5: "fig5_per_problem_deltas.png",   # Per-problem Δ histograms
    6: "fig6_v07_summary_panel.png",    # 2×3 robustness summary panel
    7: "fig7_verdict_scaling.png",      # Verdict Δ vs model parameters
    8: "fig8_per_token_surprise_example.png",  # Per-token surprise bar chart
}

for n, fn in MAPPING.items():
    dst = FIG_CACHE / f"fig{n}_panel.png"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(FIG_SRC / fn)

# Section-key -> figure number (inserted after that section's first paragraph).
# Keys match the "## N " prefix in the markdown headings.
PLAN = {
    "1 ": 1,    # Introduction -> fig1 failure matrix (hero)
    "3 ": None, # Method — no figure
    "4.2 ": 2,  # Behavioral premise -> fig2 premise summary
    "4.3 ": 3,  # Logprob measurements -> fig3 scatter
    "4.4 ": None,
    "4.6 ": 4,  # Answer-relabeled -> fig4 distribution panel
    "4.7 ": None,
    "4.8 ": 6,  # Format ablation / per-token -> fig6 summary
    "5 ": None, # Discussion on why logprob reads zero
    "6 ": 7,    # Discussion -> fig7 verdict scaling
}

_register_fonts()
out = PROJ / "paper" / "paper.pdf"
build_paper(PROJ / "paper" / "paper.md", FIG_CACHE, PLAN, out)
print("built:", out, "exists:", out.exists(), "size:", out.stat().st_size if out.exists() else 0)

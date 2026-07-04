"""Compute effect sizes (Cliff's delta), power analysis, and per-problem
delta histograms for the premise tests.

Reads the per-problem JSONLs in results/pilot/premise/ and writes:
  figures/output/effect_sizes_and_power.csv     — table with Cliff's δ + power
  figures/output/fig4_per_problem_deltas.png    — per-problem delta histograms

No Modal compute. Local CPU only.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
PREMISE = REPO_ROOT / "results" / "pilot" / "premise"
OUT = REPO_ROOT / "figures" / "output"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def cliffs_delta(x, y):
    """Cliff's δ on per-problem differences x - y. Range [-1, 1]; 0 means no
    stochastic dominance. |δ| < 0.147 = negligible; < 0.33 small;
    < 0.474 medium; else large."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n_x, n_y = x.size, y.size
    # Pairwise sign comparison
    diff = x[:, None] - y[None, :]
    n_pos = int(np.sum(diff > 0))
    n_neg = int(np.sum(diff < 0))
    return (n_pos - n_neg) / (n_x * n_y)


def cohens_d_paired(d):
    """Cohen's d on paired differences d (e.g., per-problem oracle - corrupted)."""
    d = np.asarray(d, dtype=float)
    if d.size < 2 or d.std(ddof=1) == 0:
        return float("nan")
    return float(d.mean() / d.std(ddof=1))


def power_to_detect_delta(detected_delta, sd, alpha=0.05, power=0.8):
    """Approximate two-sided paired-sample n needed to detect `detected_delta`
    with `power` at significance `alpha`. Uses the simple formula
    n = ((z_{1-α/2} + z_{1-β}) σ / Δ)^2."""
    from scipy.stats import norm
    if detected_delta == 0 or sd == 0:
        return float("inf")
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)
    return float(((z_alpha + z_beta) * sd / detected_delta) ** 2)


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


# Load all measurements
runs = {
    "behavioral_1p5b_aime":   ("p_disc", _load(PREMISE / "premise_test_1p5b.jsonl")),
    "behavioral_synthetic":   ("p_disc", _load(PREMISE / "premise_test_synthetic.jsonl")),
    "behavioral_7b_aime":     ("p_disc", _load(PREMISE / "premise_test_7b.jsonl")),
    "logprob_1p5b":           ("logp",   _load(PREMISE / "logprob_premise_1p5b.jsonl")),
    "logprob_7b":             ("logp",   _load(PREMISE / "logprob_premise_7b.jsonl")),
    # Optional Instruct files (may not exist yet)
    "behavioral_1p5b_instruct": ("p_disc", _load(PREMISE / "premise_test_1p5b_instruct.jsonl")),
    "behavioral_7b_instruct":   ("p_disc", _load(PREMISE / "premise_test_7b_instruct.jsonl")),
    "logprob_1p5b_instruct":    ("logp",   _load(PREMISE / "logprob_premise_1p5b_instruct.jsonl")),
    "logprob_7b_instruct":      ("logp",   _load(PREMISE / "logprob_premise_7b_instruct.jsonl")),
    # Llama-3.1-8B base (cross-family check)
    "behavioral_llama_3p1_8b": ("p_disc", _load(PREMISE / "premise_test_llama_3p1_8b.jsonl")),
    "logprob_llama_3p1_8b":    ("logp",   _load(PREMISE / "logprob_premise_llama_3p1_8b.jsonl")),
    # v0.6 additions: scale-out, R1-distill, cross-family
    "logprob_qwen_72b":        ("logp",   _load(PREMISE / "logprob_premise_qwen_72b.jsonl")),
    "logprob_r1_qwen_1p5b":    ("logp",   _load(PREMISE / "logprob_premise_r1_qwen_1p5b.jsonl")),
    "logprob_r1_qwen_7b":      ("logp",   _load(PREMISE / "logprob_premise_r1_qwen_7b.jsonl")),
    "logprob_r1_llama_8b":     ("logp",   _load(PREMISE / "logprob_premise_r1_llama_8b.jsonl")),
    "logprob_mistral_7b":      ("logp",   _load(PREMISE / "logprob_premise_mistral_7b.jsonl")),
    "logprob_gemma_9b":        ("logp",   _load(PREMISE / "logprob_premise_gemma_9b.jsonl")),
}

# Effect-size table
csv_path = OUT / "effect_sizes_and_power.csv"
with csv_path.open("w") as f:
    w = csv.writer(f)
    w.writerow(["run", "metric", "n", "mean_oracle", "mean_corrupted",
                "delta_mean", "delta_sd",
                "cohen_d_paired", "cliffs_delta",
                "n_for_d=0.5_at_80%power", "n_for_delta=0.1_at_80%power"])
    for name, (prefix, rows) in runs.items():
        if not rows:
            w.writerow([name, prefix, 0, "—"] + [""] * 8)
            continue
        oracle = np.array([r[f"{prefix}_oracle"] for r in rows
                          if r.get(f"{prefix}_oracle") is not None], dtype=float)
        corr = np.array([r[f"{prefix}_corrupted"] for r in rows
                        if r.get(f"{prefix}_corrupted") is not None], dtype=float)
        if oracle.size == 0 or corr.size == 0 or oracle.size != corr.size:
            w.writerow([name, prefix, len(rows), "shape mismatch"] + [""] * 8)
            continue
        deltas = oracle - corr
        d = cohens_d_paired(deltas)
        cd = cliffs_delta(oracle, corr)
        sd = float(deltas.std(ddof=1)) if deltas.size > 1 else float("nan")
        n_d_05 = power_to_detect_delta(0.5 * sd, sd)  # detect d=0.5 effect
        n_delta_pt1 = power_to_detect_delta(0.1, sd)   # detect raw delta=0.1
        w.writerow([name, prefix, oracle.size,
                    round(float(oracle.mean()), 4),
                    round(float(corr.mean()), 4),
                    round(float(deltas.mean()), 4),
                    round(sd, 4) if not np.isnan(sd) else "nan",
                    round(d, 4) if not np.isnan(d) else "nan",
                    round(cd, 4),
                    round(n_d_05, 1) if np.isfinite(n_d_05) else "inf",
                    round(n_delta_pt1, 1) if np.isfinite(n_delta_pt1) else "inf"])
print(f"wrote {csv_path}")

# ---------- Figure 4: per-problem delta histograms ------------------------
# Show all available behavioral runs in one figure, plus all logprob runs.
def _collect_deltas(prefix, rows):
    if not rows:
        return None
    o = np.array([r[f"{prefix}_oracle"]    for r in rows
                  if r.get(f"{prefix}_oracle") is not None], dtype=float)
    c = np.array([r[f"{prefix}_corrupted"] for r in rows
                  if r.get(f"{prefix}_corrupted") is not None], dtype=float)
    if o.size == 0 or o.size != c.size:
        return None
    return o - c


beh_panels = [
    ("AIME H_K (Qwen 1.5B base)",     _collect_deltas("p_disc", runs["behavioral_1p5b_aime"][1])),
    ("Synthetic (Qwen 1.5B base)",    _collect_deltas("p_disc", runs["behavioral_synthetic"][1])),
    ("AIME H_K (Qwen 7B base)",       _collect_deltas("p_disc", runs["behavioral_7b_aime"][1])),
    ("AIME H_K (Qwen 1.5B Instruct)", _collect_deltas("p_disc", runs["behavioral_1p5b_instruct"][1])),
    ("AIME H_K (Qwen 7B Instruct)",   _collect_deltas("p_disc", runs["behavioral_7b_instruct"][1])),
    ("AIME H_K (Llama-3.1-8B base)",  _collect_deltas("p_disc", runs["behavioral_llama_3p1_8b"][1])),
]
lp_panels = [
    ("AIME H_K (Qwen 1.5B base) logp",     _collect_deltas("logp", runs["logprob_1p5b"][1])),
    ("AIME H_K (Qwen 7B base) logp",       _collect_deltas("logp", runs["logprob_7b"][1])),
    ("AIME H_K (Qwen 72B base) logp",      _collect_deltas("logp", runs["logprob_qwen_72b"][1])),
    ("AIME H_K (Qwen 1.5B Instruct) logp", _collect_deltas("logp", runs["logprob_1p5b_instruct"][1])),
    ("AIME H_K (Qwen 7B Instruct) logp",   _collect_deltas("logp", runs["logprob_7b_instruct"][1])),
    ("AIME H_K (Llama-3.1-8B base) logp",  _collect_deltas("logp", runs["logprob_llama_3p1_8b"][1])),
    ("AIME H_K (Mistral-7B base) logp",    _collect_deltas("logp", runs["logprob_mistral_7b"][1])),
    ("AIME H_K (Gemma-2-9B base) logp",    _collect_deltas("logp", runs["logprob_gemma_9b"][1])),
    ("AIME H_K (R1-Distill-Qwen-1.5B) logp", _collect_deltas("logp", runs["logprob_r1_qwen_1p5b"][1])),
    ("AIME H_K (R1-Distill-Qwen-7B) logp",   _collect_deltas("logp", runs["logprob_r1_qwen_7b"][1])),
    ("AIME H_K (R1-Distill-Llama-8B) logp",  _collect_deltas("logp", runs["logprob_r1_llama_8b"][1])),
]
beh_panels = [(t, d) for t, d in beh_panels if d is not None]
lp_panels = [(t, d) for t, d in lp_panels if d is not None]

n_panels = len(beh_panels) + len(lp_panels)
n_cols = 4
n_rows = (n_panels + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols,
                         figsize=(3.0 * n_cols, 2.4 * n_rows),
                         squeeze=False)

panels = beh_panels + lp_panels
for i, (title, d) in enumerate(panels):
    ax = axes[i // n_cols, i % n_cols]
    is_logp = " logp" in title
    bins = np.linspace(min(d.min(), -max(abs(d.min()), abs(d.max()))),
                       max(d.max(),  max(abs(d.min()), abs(d.max()))),
                       21)
    ax.hist(d, bins=bins, color="#1f77b4", alpha=0.85, edgecolor="black",
            linewidth=0.4)
    ax.axvline(0, color="grey", lw=0.8, ls="--")
    ax.axvline(d.mean(), color="#d62728", lw=1.2,
                label=f"mean={d.mean():.4f}")
    n = d.size
    n_pos = int(np.sum(d > 0))
    n_neg = int(np.sum(d < 0))
    n_zero = n - n_pos - n_neg
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(r"per-problem $\Delta$" + (" (logprob)" if is_logp else ""),
                  fontsize=8)
    ax.set_ylabel("count", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.text(0.02, 0.97,
             f"+:{n_pos}, 0:{n_zero}, –:{n_neg}\nn={n}",
             transform=ax.transAxes, fontsize=7, va="top",
             bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="grey",
                       alpha=0.85))
    ax.legend(loc="upper right", fontsize=7, frameon=False)

# Hide unused axes
for j in range(n_panels, n_rows * n_cols):
    axes[j // n_cols, j % n_cols].axis("off")

fig.suptitle("Per-problem $\\Delta = $ score(oracle) $-$ score(corrupted).\n"
             "If the asymmetry held, mass would shift right of zero.",
             fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "fig5_per_problem_deltas.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'fig5_per_problem_deltas.png'}")
print(f"panels rendered: {n_panels} (will grow when Instruct JSONLs land)")

print("done")

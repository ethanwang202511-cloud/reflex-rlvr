"""Render the paper figures from the per-problem JSONL results.

Reads (graceful — missing files just skip the panel):
  results/pilot/premise/premise_test_*.jsonl     behavioral premise results
  results/pilot/premise/logprob_premise_*.jsonl  logprob premise results

Writes (figures/output/):
  fig1_failure_matrix.png      HERO — forest plot of Delta across every
                               measurement, with the pre-registered
                               Delta>=0.2 threshold line.
  fig2_premise_summary.png     bar chart of all conditions (was fig1).
  fig3_logprob_scatter.png     per-problem logp(oracle) vs logp(corrupted).
  fig4_logprob_distribution.png paired box+strip per-token logp distributions.

Tabular data + per-problem 95% bootstrap CIs are written alongside as CSV.
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


def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _bootstrap_ci(values: list[float], n_boot: int = 5000, alpha: float = 0.05,
                  seed: int = 1337) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _bootstrap_ci_delta(rows, prefix, n_boot=5000, alpha=0.05, seed=1337):
    """95% bootstrap CI on the per-problem mean of (oracle - corrupted)."""
    if not rows:
        return float("nan"), float("nan"), float("nan")
    o = np.array([r[f"{prefix}_oracle"] for r in rows
                  if r.get(f"{prefix}_oracle") is not None], dtype=float)
    c = np.array([r[f"{prefix}_corrupted"] for r in rows
                  if r.get(f"{prefix}_corrupted") is not None], dtype=float)
    if o.size == 0 or c.size != o.size:
        return float("nan"), float("nan"), float("nan")
    deltas = o - c
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, deltas.size, size=(n_boot, deltas.size))
    means = deltas[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(deltas.mean()), float(lo), float(hi)


# ---------- Load all results ----------------------------------------------
beh_1p5b_aime = _load(PREMISE / "premise_test_1p5b.jsonl")
beh_synthetic = _load(PREMISE / "premise_test_synthetic.jsonl")
beh_7b_aime = _load(PREMISE / "premise_test_7b.jsonl")
beh_1p5b_inst = _load(PREMISE / "premise_test_1p5b_instruct.jsonl")
beh_7b_inst = _load(PREMISE / "premise_test_7b_instruct.jsonl")
beh_llama = _load(PREMISE / "premise_test_llama_3p1_8b.jsonl")

lp_1p5b = _load(PREMISE / "logprob_premise_1p5b.jsonl")
lp_7b = _load(PREMISE / "logprob_premise_7b.jsonl")
lp_1p5b_inst = _load(PREMISE / "logprob_premise_1p5b_instruct.jsonl")
lp_7b_inst = _load(PREMISE / "logprob_premise_7b_instruct.jsonl")
lp_llama = _load(PREMISE / "logprob_premise_llama_3p1_8b.jsonl")

# v0.6 additions: scale, R1-distill, cross-family
lp_qwen_72b = _load(PREMISE / "logprob_premise_qwen_72b.jsonl")
lp_r1_qwen_1p5b = _load(PREMISE / "logprob_premise_r1_qwen_1p5b.jsonl")
lp_r1_qwen_7b = _load(PREMISE / "logprob_premise_r1_qwen_7b.jsonl")
lp_r1_llama_8b = _load(PREMISE / "logprob_premise_r1_llama_8b.jsonl")
lp_mistral_7b = _load(PREMISE / "logprob_premise_mistral_7b.jsonl")
lp_gemma_9b = _load(PREMISE / "logprob_premise_gemma_9b.jsonl")


def _means_with_ci(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    mean = float(np.mean(vals)) if vals else float("nan")
    lo, hi = _bootstrap_ci(vals)
    return mean, lo, hi


# ---------- Tabular CSV output (paper-quality bootstrap CIs) ---------------
all_runs = [
    # (label, rows, prefix, family)
    ("AIME H_K (Qwen2.5-1.5B base)",                beh_1p5b_aime, "p_disc", "Qwen base"),
    ("Synthetic (Qwen2.5-1.5B base)",               beh_synthetic, "p_disc", "Qwen base"),
    ("AIME H_K (Qwen2.5-7B base)",                  beh_7b_aime,   "p_disc", "Qwen base"),
    ("AIME H_K (Qwen2.5-1.5B-Instruct)",            beh_1p5b_inst, "p_disc", "Qwen Instruct"),
    ("AIME H_K (Qwen2.5-7B-Instruct)",              beh_7b_inst,   "p_disc", "Qwen Instruct"),
    ("AIME H_K (Llama-3.1-8B base)",                beh_llama,     "p_disc", "Llama"),
    ("AIME H_K (Qwen2.5-1.5B base) logprob",        lp_1p5b,       "logp",   "Qwen base"),
    ("AIME H_K (Qwen2.5-7B base) logprob",          lp_7b,         "logp",   "Qwen base"),
    ("AIME H_K (Qwen2.5-72B base) logprob",         lp_qwen_72b,   "logp",   "Qwen base 72B"),
    ("AIME H_K (Qwen2.5-1.5B-Instruct) logprob",    lp_1p5b_inst,  "logp",   "Qwen Instruct"),
    ("AIME H_K (Qwen2.5-7B-Instruct) logprob",      lp_7b_inst,    "logp",   "Qwen Instruct"),
    ("AIME H_K (Llama-3.1-8B base) logprob",        lp_llama,      "logp",   "Llama"),
    ("AIME H_K (Mistral-7B-v0.3 base) logprob",     lp_mistral_7b, "logp",   "Mistral"),
    ("AIME H_K (Gemma-2-9B base) logprob",          lp_gemma_9b,   "logp",   "Gemma"),
    ("AIME H_K (R1-Distill-Qwen-1.5B) logprob",     lp_r1_qwen_1p5b, "logp", "R1-Distill"),
    ("AIME H_K (R1-Distill-Qwen-7B) logprob",       lp_r1_qwen_7b,   "logp", "R1-Distill"),
    ("AIME H_K (R1-Distill-Llama-8B) logprob",      lp_r1_llama_8b,  "logp", "R1-Distill"),
]

csv_path = OUT / "premise_results_table.csv"
with csv_path.open("w") as f:
    w = csv.writer(f)
    w.writerow(["dataset", "metric", "no_cot", "no_cot_ci_lo", "no_cot_ci_hi",
                "oracle", "oracle_ci_lo", "oracle_ci_hi",
                "corrupted", "corrupted_ci_lo", "corrupted_ci_hi",
                "shuffled", "shuffled_ci_lo", "shuffled_ci_hi", "n"])
    for label, rows, prefix, family in all_runs:
        if not rows:
            w.writerow([label, prefix, ""] * 1 + ["pending"] + [""] * 12 + [0])
            continue
        out_row = [label, prefix]
        for cond in ("no_cot", "oracle", "corrupted", "shuffled"):
            mean, lo, hi = _means_with_ci(rows, f"{prefix}_{cond}")
            out_row += [round(mean, 4), round(lo, 4), round(hi, 4)]
        out_row.append(len(rows))
        w.writerow(out_row)
print(f"wrote {csv_path}")


# ---------- Figure 1: HERO Failure Matrix forest plot ----------------------
# A horizontal forest plot of the Delta = oracle - corrupted across every
# measurement, with 95% bootstrap CIs and the pre-registered Delta>=0.2
# threshold drawn as a vertical line.
fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.5),
                          gridspec_kw={"width_ratios": [1, 1]})
ax_b, ax_l = axes

family_colors = {
    "Qwen base":      "#1f77b4",
    "Qwen base 72B":  "#08306b",
    "Qwen Instruct":  "#ff7f0e",
    "Llama":          "#2ca02c",
    "Mistral":        "#9467bd",
    "Gemma":          "#8c564b",
    "R1-Distill":     "#d62728",
}

beh_runs = [(label, rows, prefix, family)
            for label, rows, prefix, family in all_runs if prefix == "p_disc"]
lp_runs = [(label, rows, prefix, family)
           for label, rows, prefix, family in all_runs if prefix == "logp"]


def _short_label(label: str) -> str:
    """Convert long descriptive label to compact y-axis label."""
    return (label
            .replace("AIME H_K ", "")
            .replace("Synthetic ", "synth: ")
            .replace(" logprob", "")
            .replace("(", "").replace(")", ""))


def _forest(ax, runs, x_threshold, x_label, title):
    n = len(runs)
    y = np.arange(n)
    deltas, los, his, colors, labels, has_data = [], [], [], [], [], []
    for label, rows, prefix, family in runs:
        d, lo, hi = _bootstrap_ci_delta(rows, prefix)
        deltas.append(d if not np.isnan(d) else 0.0)
        los.append(lo if not np.isnan(lo) else 0.0)
        his.append(hi if not np.isnan(hi) else 0.0)
        colors.append(family_colors.get(family, "#666666"))
        labels.append(_short_label(label))
        has_data.append(not np.isnan(d))
    deltas = np.asarray(deltas, dtype=float)
    los = np.asarray(los, dtype=float)
    his = np.asarray(his, dtype=float)
    err = np.array([deltas - los, his - deltas])
    err = np.clip(err, 0.0, None)  # bootstrap can flip due to numerics
    for i in range(n):
        if has_data[i]:
            ax.errorbar(deltas[i], y[i], xerr=[[err[0, i]], [err[1, i]]],
                        fmt="o", color=colors[i], ecolor=colors[i],
                        elinewidth=1.6, markersize=7, capsize=4)
        else:
            ax.text(0, y[i], "  pending", color="grey", fontsize=8, va="center")
    ax.axvline(0, color="black", lw=0.8, alpha=0.7)
    if x_threshold is not None:
        ax.axvline(x_threshold, color="#d62728", lw=1.2, ls="--",
                   alpha=0.85)
        ax.text(x_threshold, n - 0.4,
                f"  pre-reg threshold $\\Delta\\geq{x_threshold}$",
                color="#d62728", fontsize=9, va="top")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(x_label, fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="x", lw=0.3, alpha=0.4)
    return deltas, los, his


_forest(ax_b, beh_runs, x_threshold=0.2,
        x_label=r"$\Delta = p_{\rm disc}({\rm oracle}) - p_{\rm disc}({\rm corrupted})$",
        title=f"Behavioral premise test ({len(beh_runs)} measurements)")
_forest(ax_l, lp_runs, x_threshold=None,
        x_label=r"$\Delta\log p = \log p({\rm answer}\mid{\rm oracle}) - \log p({\rm answer}\mid{\rm corrupted})$",
        title=f"Logprob premise test ({len(lp_runs)} measurements)")

# legend (family)
handles = [plt.Line2D([0], [0], marker="o", color=c, lw=0, label=fam,
                       markersize=7)
           for fam, c in family_colors.items()]
fig.legend(handles=handles, loc="lower center", ncol=len(family_colors),
           frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))

fig.suptitle(
    "Failure Matrix. $\\Delta\\approx 0$ across every base, Instruct, R1-distilled, and cross-family measurement\n"
    "(Qwen 1.5B–72B, Llama, Mistral, Gemma, R1-Distill family).\n"
    "The pre-registered $\\Delta\\geq 0.2$ behavioral threshold (red dashed) is never approached.",
    fontsize=11, y=1.02)
fig.tight_layout()
fig.savefig(OUT / "fig1_failure_matrix.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'fig1_failure_matrix.png'}")


# ---------- Figure 2: bar chart of behavioral / logprob measurements -------
# (was fig1 in v0.5; still useful for showing absolute values + shuffled cue)
fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.0),
                          gridspec_kw={"width_ratios": [5, 6]})
ax_b, ax_l = axes

conds = ["no_cot", "oracle", "corrupted", "shuffled"]
cond_labels = ["no\nCoT", "oracle", "corrupted", "shuffled"]
cond_colors = ["#bbbbbb", "#1f77b4", "#d62728", "#2ca02c"]


def _short(label):
    return (label
            .replace("AIME H_K (", "")
            .replace("Synthetic (", "synth ")
            .replace(") logprob", "")
            .replace(")", "")
            .replace("Qwen2.5-", "Q ")
            .replace("Llama-3.1-8B", "Llama 8B")
            .replace("Mistral-7B-v0.3", "Mistral 7B")
            .replace("Gemma-2-9B", "Gemma 9B")
            .replace("R1-Distill-", "R1d "))


# Behavioral panel
beh_panels_runs = [r for r in all_runs if r[2] == "p_disc"]
x_centers = np.arange(len(beh_panels_runs))
bar_w = 0.18
for j, cond in enumerate(conds):
    means, los, his = [], [], []
    for label, rows, prefix, family in beh_panels_runs:
        if rows:
            m, lo, hi = _means_with_ci(rows, f"{prefix}_{cond}")
        else:
            m = lo = hi = float("nan")
        means.append(m); los.append(lo); his.append(hi)
    means_arr = np.asarray(means, dtype=float)
    err_lo = np.where(np.isnan(means_arr), 0,
                       np.nan_to_num(means_arr - np.asarray(los, dtype=float),
                                     nan=0.0))
    err_hi = np.where(np.isnan(means_arr), 0,
                       np.nan_to_num(np.asarray(his, dtype=float) - means_arr,
                                     nan=0.0))
    means_plot = np.nan_to_num(means_arr, nan=0.0)
    ax_b.bar(x_centers + (j - 1.5) * bar_w, means_plot, bar_w,
             yerr=[err_lo, err_hi], label=cond_labels[j],
             color=cond_colors[j], capsize=2)
ax_b.set_xticks(x_centers)
ax_b.set_xticklabels([_short(r[0]) for r in beh_panels_runs],
                     fontsize=7, rotation=20, ha="right")
ax_b.set_ylabel(r"$p_{\rm disc}$ (behavioral)", fontsize=10)
ax_b.set_ylim(0, max(0.85, ax_b.get_ylim()[1]))
ax_b.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.5)
ax_b.set_title("Behavioral premise test (4 conditions × 8 samples)",
               fontsize=10)
ax_b.legend(loc="upper right", fontsize=8, frameon=False, ncol=2)

# Logprob panel
lp_panels_runs = [r for r in all_runs if r[2] == "logp"]
x_centers = np.arange(len(lp_panels_runs))
for j, cond in enumerate(conds):
    means, los, his = [], [], []
    for label, rows, prefix, family in lp_panels_runs:
        if rows:
            m, lo, hi = _means_with_ci(rows, f"{prefix}_{cond}")
        else:
            m = lo = hi = float("nan")
        means.append(m); los.append(lo); his.append(hi)
    means_arr = np.asarray(means, dtype=float)
    err_lo = np.where(np.isnan(means_arr), 0,
                       np.nan_to_num(means_arr - np.asarray(los, dtype=float),
                                     nan=0.0))
    err_hi = np.where(np.isnan(means_arr), 0,
                       np.nan_to_num(np.asarray(his, dtype=float) - means_arr,
                                     nan=0.0))
    means_plot = np.nan_to_num(means_arr, nan=0.0)
    ax_l.bar(x_centers + (j - 1.5) * bar_w, means_plot, bar_w,
             yerr=[err_lo, err_hi], color=cond_colors[j], capsize=2)
ax_l.set_xticks(x_centers)
ax_l.set_xticklabels([_short(r[0]) for r in lp_panels_runs],
                     fontsize=7, rotation=20, ha="right")
ax_l.set_ylabel(r"mean per-token $\log P({\rm answer}\mid{\rm prompt})$",
                fontsize=10)
ax_l.set_title("Logprob premise test", fontsize=10)

fig.suptitle("Absolute scores across every measurement. Oracle and corrupted columns are visually identical at every setting;\n"
             "the no_cot baseline is uniformly worse, confirming the model DOES condition on the prefix — just not on its semantics.",
             fontsize=11, y=1.04)
fig.tight_layout()
fig.savefig(OUT / "fig2_premise_summary.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'fig2_premise_summary.png'}")


# ---------- Figure 3: scatter of logp(oracle) vs logp(corrupted) -----------
lp_for_scatter = [(label, rows) for label, rows, prefix, family in all_runs
                   if prefix == "logp" and rows]
n_panels = len(lp_for_scatter)
n_cols = 4
n_rows = (n_panels + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.4 * n_rows),
                         squeeze=False)
for idx, (label, rows) in enumerate(lp_for_scatter):
    ax = axes[idx // n_cols, idx % n_cols]
    o = np.array([r["logp_oracle"] for r in rows], dtype=float)
    c = np.array([r["logp_corrupted"] for r in rows], dtype=float)
    if o.size == 0:
        ax.axis("off"); continue
    lo = float(min(o.min(), c.min())) - 0.1
    hi = float(max(o.max(), c.max())) + 0.1
    ax.plot([lo, hi], [lo, hi], color="grey", lw=0.8, ls="--",
            label="oracle = corrupted")
    ax.scatter(c, o, s=14, alpha=0.7, color="#1f77b4",
               edgecolor="black", linewidth=0.4)
    n_above = int(np.sum(o > c))
    short = _short(label)
    ax.set_xlabel(r"$\log P({\rm answer}\mid x, y_{\rm corrupted})$", fontsize=8)
    ax.set_ylabel(r"$\log P({\rm answer}\mid x, y_{\rm oracle})$", fontsize=8)
    ax.set_title(f"{short}\n({n_above}/{len(rows)} oracle>corrupted)",
                 fontsize=9)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.tick_params(labelsize=7)
    ax.legend(loc="upper left", fontsize=7, frameon=False)

# Hide unused axes
for j in range(n_panels, n_rows * n_cols):
    axes[j // n_cols, j % n_cols].axis("off")

fig.suptitle("Per-problem logprob: oracle vs corrupted chains across every setting.\n"
             "Points cluster on the y=x diagonal in every panel.",
             fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "fig3_logprob_scatter.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'fig3_logprob_scatter.png'}")


# ---------- Figure 4: per-token logp distributions box+strip ---------------
lp_for_dist = [(label, rows) for label, rows, prefix, family in all_runs
                if prefix == "logp" and rows]
n_panels = len(lp_for_dist)
n_cols = 4
n_rows = (n_panels + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.0 * n_rows),
                         squeeze=False)
for idx, (label, rows) in enumerate(lp_for_dist):
    ax = axes[idx // n_cols, idx % n_cols]
    if not rows:
        ax.axis("off"); continue
    data = [
        np.array([r["logp_no_cot"]    for r in rows], dtype=float),
        np.array([r["logp_oracle"]    for r in rows], dtype=float),
        np.array([r["logp_corrupted"] for r in rows], dtype=float),
        np.array([r["logp_shuffled"]  for r in rows], dtype=float),
    ]
    ax.boxplot(data, positions=range(len(data)), widths=0.5,
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor="white", edgecolor="black"),
                medianprops=dict(color="#d62728", lw=1.4))
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        jitter = rng.uniform(-0.12, 0.12, size=d.size)
        ax.scatter(np.full_like(d, i, dtype=float) + jitter, d,
                   s=8, alpha=0.5, color=cond_colors[i], edgecolor="none")
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(["no_cot", "oracle", "corr.", "shuf."], fontsize=8)
    ax.set_ylabel("per-token logprob", fontsize=8)
    ax.set_title(_short(label), fontsize=9)
    ax.tick_params(labelsize=7)

for j in range(n_panels, n_rows * n_cols):
    axes[j // n_cols, j % n_cols].axis("off")

fig.suptitle("Per-problem distribution of mean per-token log P(answer | prompt)\n"
             "across 4 conditions, every setting.",
             fontsize=11, y=1.01)
fig.tight_layout()
fig.savefig(OUT / "fig4_logprob_distribution.png", dpi=200,
            bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT / 'fig4_logprob_distribution.png'}")

print("done")

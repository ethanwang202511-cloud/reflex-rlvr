"""Regenerate Figure 1 — vector PDF, clean text."""
from __future__ import annotations
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "results" / "pilot" / "premise" / "analysis"
OUT_PDF = ROOT / "figures" / "output" / "fig1_failure_matrix.pdf"
OUT_PNG = ROOT / "figures" / "output" / "fig1_failure_matrix.png"

def _is_canonical_row(r):
    # Drop seed-variant and SFT slugs (e.g. "qwen_7b_instruct_seed37", "llama_3p1_8b_sft").
    s = r.get("slug", "")
    return "_seed" not in s and not s.endswith("_sft")

cc = [r for r in csv.DictReader(open(ANALYSIS / "consistent_corruption_summary.csv")) if _is_canonical_row(r)]
v = [r for r in csv.DictReader(open(ANALYSIS / "verdict_summary.csv")) if _is_canonical_row(r)]

def sort_key(m):
    if m.startswith("Qwen") and "Instruct" not in m: return (0, m)
    if m.startswith("Qwen") and "Instruct" in m: return (1, m)
    if "DeepSeek-R1" in m: return (2, m)
    return (3, m)

models = sorted({r["model"] for r in cc} | {r["model"] for r in v}, key=sort_key)
cc_map = {r["model"]: r for r in cc}
v_map = {r["model"]: r for r in v}

fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharey=True)

ax = axes[0]
y = range(len(models))
for i, m in enumerate(models):
    if m in cc_map:
        r = cc_map[m]
        d = float(r["mean_disc_signal"])
        lo = float(r["disc_signal_ci_lo"]); hi = float(r["disc_signal_ci_hi"])
        ax.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="s",
                    color="#1f77b4" if r["gate_passed"] == "True" else "#d62728",
                    markersize=7, capsize=3, markeredgecolor="black", markeredgewidth=0.5)
        ax.text(d + 0.06, i, f"{d:.2f}", va="center", fontsize=9)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(list(y))
ax.set_yticklabels(models, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel(r"$\sigma_\mathrm{disc}$ (logp units)", fontsize=10)
ax.set_title("Probe 1: answer-relabeled corruption\n(every model passes; old probe gave $\\Delta\\approx0$)", fontsize=10)
ax.grid(axis="x", linestyle=":", alpha=0.5)
ax.set_xlim(-0.5, 4.0)

ax = axes[1]
for i, m in enumerate(models):
    if m in v_map:
        r = v_map[m]
        d = float(r["mean_delta_verdict"])
        lo = float(r["delta_ci_lo"]); hi = float(r["delta_ci_hi"])
        ax.errorbar(d, i, xerr=[[d - lo], [hi - d]], fmt="o",
                    color="#1f77b4" if r["gate_passed"] == "True" else "#d62728",
                    markersize=7, capsize=3, markeredgecolor="black", markeredgewidth=0.5)
        ax.text(d + 0.15, i, f"{d:+.2f}", va="center", fontsize=9)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel(r"$\Delta_\mathrm{verdict}$ (logit units)", fontsize=10)
ax.set_title("Probe 2: YES/NO verdict\n(3 non-Qwen base models fail despite Probe 1 passing)", fontsize=10)
ax.grid(axis="x", linestyle=":", alpha=0.5)
ax.set_xlim(-0.7, 9.5)

blue = mpatches.Patch(color="#1f77b4", label="Gate PASSED")
red = mpatches.Patch(color="#d62728", label="Gate FAILED")
fig.legend(handles=[blue, red], loc="upper center", ncol=2,
           frameon=False, bbox_to_anchor=(0.5, 1.00))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
print(f"Wrote {OUT_PDF}\nWrote {OUT_PNG}")

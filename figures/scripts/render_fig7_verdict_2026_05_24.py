"""Regenerate Figure 7 (verdict scaling) — vector PDF, clean text."""
from __future__ import annotations
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Force vector text + TrueType font embedding (no Type 3 paths, no glyph subsetting weirdness)
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

ROOT = Path(__file__).resolve().parents[2]
CSV = ROOT / "results" / "pilot" / "premise" / "analysis" / "verdict_summary.csv"
OUT_PDF = ROOT / "figures" / "output" / "fig7_verdict_scaling.pdf"
OUT_PNG = ROOT / "figures" / "output" / "fig7_verdict_scaling.png"

def _is_canonical_row(r):
    s = r.get("slug", "")
    return "_seed" not in s and not s.endswith("_sft")

rows = [r for r in csv.DictReader(open(CSV)) if _is_canonical_row(r)]

def sort_key(r):
    m = r["model"]
    if m.startswith("Qwen") and "Instruct" not in m: return (0, m)
    if m.startswith("Qwen") and "Instruct" in m: return (1, m)
    if "DeepSeek-R1" in m: return (2, m)
    return (3, m)
rows.sort(key=sort_key)

models = [r["model"] for r in rows]
deltas = [float(r["mean_delta_verdict"]) for r in rows]
lo = [float(r["delta_ci_lo"]) for r in rows]
hi = [float(r["delta_ci_hi"]) for r in rows]
gate = [r["gate_passed"] == "True" for r in rows]
errs_lo = [d - l for d, l in zip(deltas, lo)]
errs_hi = [h - d for d, h in zip(deltas, hi)]

fig, ax = plt.subplots(figsize=(8.5, 5.0))
y = list(range(len(models)))
colors = ["#1f77b4" if g else "#d62728" for g in gate]
ax.barh(y, deltas, xerr=[errs_lo, errs_hi], color=colors,
        edgecolor="black", linewidth=0.6, capsize=3)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_yticks(y)
ax.set_yticklabels(models, fontsize=10)
ax.invert_yaxis()
ax.set_xlabel(r"$\Delta_\mathrm{verdict}$ = $\log p(\mathrm{YES}\,|\,\mathrm{oracle}) - \log p(\mathrm{YES}\,|\,\mathrm{corrupted})$  (95% CI)", fontsize=10)
n_total = len(rows)
n_pass = sum(gate)
n_fail = n_total - n_pass
ax.set_title(f"Probe 2: YES/NO verdict discrimination across {n_total} model variants",
             fontsize=11, pad=12)
ax.grid(axis="x", linestyle=":", alpha=0.5)

for i, (d, g) in enumerate(zip(deltas, gate)):
    label = "PASS" if g else "FAIL"
    offset = 0.18 if d > 0 else -0.18
    ax.text(d + offset, i, f"{d:+.2f}  {label}", va="center",
            ha="left" if d > 0 else "right", fontsize=9)

blue = mpatches.Patch(color="#1f77b4", label=f"Gate PASSED ({n_pass} / {n_total})")
red = mpatches.Patch(color="#d62728", label=f"Gate FAILED ({n_fail} / {n_total})")
ax.legend(handles=[blue, red], loc="lower right", fontsize=9, frameon=True)

ax.set_xlim(min(deltas + [0]) - 0.5, max(hi) + 1.5)
plt.tight_layout()
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
print(f"Wrote {OUT_PDF}\nWrote {OUT_PNG}")

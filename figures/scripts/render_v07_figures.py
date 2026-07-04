"""Render the v0.7 experiment figures.

Reads (graceful — missing files skip that data point, never crash):
  results/pilot/premise/consistent_corruption_*.jsonl
  results/pilot/premise/verdict_*.jsonl
  results/pilot/premise/format_{chat,interleaved}_qwen_1p5b.jsonl
  results/pilot/premise/per_token_surprise_qwen_1p5b.jsonl
  results/pilot/premise/logprob_premise_1p5b.jsonl   (for cross-scatter)

Writes (figures/output/):
  fig6_v07_summary_panel.png   2×3 summary panel
  fig7_verdict_scaling.png     scaling scatter+line
  fig8_per_token_surprise_example.png  per-token example (skipped if no series)
"""
from __future__ import annotations

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

# ── consistent model-level palette ──────────────────────────────────────────
MODEL_COLOR = {
    "1.5B":        "#1f77b4",   # blue
    "7B":          "#2ca02c",   # green
    "7B-Instruct": "#d62728",   # red
    "72B":         "#9467bd",   # purple
    "R1-1.5B":     "#ff7f0e",   # orange
}


# ── helpers ──────────────────────────────────────────────────────────────────
def _load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _bootstrap_ci(values: list[float], n_boot: int = 5000, alpha: float = 0.05,
                  seed: int = 1337) -> tuple[float, float, float]:
    """Return (mean, lo_95, hi_95) via bootstrap resampling."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(boot_means, [alpha / 2, 1 - alpha / 2])
    return mean, float(lo), float(hi)


def _bar_with_ci(ax, x, mean, lo, hi, color, width=0.6, **kwargs):
    """Draw a single bar with 95% CI errorbars."""
    if np.isnan(mean):
        ax.text(x, 0.02, "pending", ha="center", va="bottom",
                fontsize=8, color="grey", rotation=90)
        return
    err_lo = max(0.0, mean - lo)
    err_hi = max(0.0, hi - mean)
    ax.bar(x, mean, width, color=color, alpha=0.85, **kwargs)
    ax.errorbar(x, mean, yerr=[[err_lo], [err_hi]],
                fmt="none", color="black", elinewidth=1.4, capsize=4)


# ── load data ────────────────────────────────────────────────────────────────
cc_1p5b   = _load(PREMISE / "consistent_corruption_qwen_1p5b.jsonl")
cc_7b     = _load(PREMISE / "consistent_corruption_qwen_7b.jsonl")
cc_7b_ins = _load(PREMISE / "consistent_corruption_qwen_7b_instruct.jsonl")
cc_r1     = _load(PREMISE / "consistent_corruption_r1_qwen_1p5b.jsonl")
cc_72b    = _load(PREMISE / "consistent_corruption_qwen_72b.jsonl")   # may be empty

vd_1p5b   = _load(PREMISE / "verdict_qwen_1p5b.jsonl")
vd_7b     = _load(PREMISE / "verdict_qwen_7b.jsonl")
vd_7b_ins = _load(PREMISE / "verdict_qwen_7b_instruct.jsonl")
vd_72b    = _load(PREMISE / "verdict_qwen_72b.jsonl")
vd_r1     = _load(PREMISE / "verdict_r1_qwen_1p5b.jsonl")

fmt_chat  = _load(PREMISE / "format_chat_qwen_1p5b.jsonl")
fmt_int   = _load(PREMISE / "format_interleaved_qwen_1p5b.jsonl")

pts       = _load(PREMISE / "per_token_surprise_qwen_1p5b.jsonl")
lp_1p5b   = _load(PREMISE / "logprob_premise_1p5b.jsonl")


# ── Figure 6: 2×3 summary panel ──────────────────────────────────────────────
fig6, axes = plt.subplots(2, 3, figsize=(16, 10))
fig6.suptitle(
    "v0.7 experiment battery — three independent corruption-detection tests",
    fontsize=13, y=1.01,
)

# ── Panel A (top-left): consistent corruption discrimination signal ──────────
ax = axes[0, 0]
cc_models = [
    ("1.5B",        cc_1p5b,   "Qwen-1.5B base"),
    ("7B",          cc_7b,     "Qwen-7B base"),
    ("7B-Instruct", cc_7b_ins, "Qwen-7B-Inst."),
    ("R1-1.5B",     cc_r1,     "R1-Distill-1.5B"),
    ("72B",         cc_72b,    "Qwen-72B base"),
]
for xi, (key, rows, label) in enumerate(cc_models):
    vals = [r["discrimination_signal"] for r in rows
            if r.get("discrimination_signal") is not None]
    mean, lo, hi = _bootstrap_ci(vals)
    _bar_with_ci(ax, xi, mean, lo, hi, color=MODEL_COLOR[key])
ax.set_xticks(range(len(cc_models)))
ax.set_xticklabels([m[2] for m in cc_models], fontsize=8, rotation=18, ha="right")
ax.set_ylabel("discrimination signal (logp)", fontsize=9)
ax.axhline(0, color="black", lw=0.8, alpha=0.6)
ax.set_title("Answer-relabeled corruption (§4.8.1)", fontsize=10)
ax.grid(axis="y", lw=0.3, alpha=0.4)

# ── Panel B (top-middle): verdict Δ logit across models ─────────────────────
ax = axes[0, 1]
vd_models = [
    ("1.5B",        vd_1p5b,   "Qwen-1.5B base"),
    ("7B",          vd_7b,     "Qwen-7B base"),
    ("7B-Instruct", vd_7b_ins, "Qwen-7B-Inst."),
    ("72B",         vd_72b,    "Qwen-72B base"),
    ("R1-1.5B",     vd_r1,     "R1-Distill-1.5B"),
]
for xi, (key, rows, label) in enumerate(vd_models):
    vals = [r["delta_oracle_minus_corrupted"] for r in rows
            if r.get("delta_oracle_minus_corrupted") is not None]
    mean, lo, hi = _bootstrap_ci(vals)
    _bar_with_ci(ax, xi, mean, lo, hi, color=MODEL_COLOR[key])
ax.set_xticks(range(len(vd_models)))
ax.set_xticklabels([m[2] for m in vd_models], fontsize=8, rotation=18, ha="right")
ax.set_ylabel("Δ verdict logit (YES−NO, oracle−corrupted)", fontsize=9)
ax.axhline(0, color="black", lw=0.8, alpha=0.6)
ax.set_title("Verdict (YES/NO) test (§4.8.2)", fontsize=10)
ax.grid(axis="y", lw=0.3, alpha=0.4)

# ── Panel C (top-right): v0.6 logp Δ vs v0.7 verdict Δ ─────────────────────
ax = axes[0, 2]
# v0.6 hard-coded means (from logprob_premise results); all near zero = fail
v06_vals = {
    "1.5B":        -0.0024,
    "7B":          -0.0071,
    "7B-Instruct": -0.0079,
    "R1-1.5B":     -0.0120,
    "72B":          -0.0194,
}
# v0.7 verdict Δ means from data
v07_vals: dict[str, float] = {}
for key, rows, _label in vd_models:
    vals = [r["delta_oracle_minus_corrupted"] for r in rows
            if r.get("delta_oracle_minus_corrupted") is not None]
    v07_vals[key] = float(np.mean(vals)) if vals else float("nan")

bar_w = 0.35
models_compare = ["1.5B", "7B", "7B-Instruct", "R1-1.5B", "72B"]
x = np.arange(len(models_compare))
for xi, key in enumerate(models_compare):
    v06 = v06_vals.get(key, float("nan"))
    v07 = v07_vals.get(key, float("nan"))
    c = MODEL_COLOR[key]
    if not np.isnan(v06):
        ax.bar(xi - bar_w / 2, v06, bar_w, color=c, alpha=0.45,
               label="v0.6 logp Δ" if xi == 0 else "_nolegend_",
               hatch="//", edgecolor="black", linewidth=0.5)
    if not np.isnan(v07):
        ax.bar(xi + bar_w / 2, v07, bar_w, color=c, alpha=0.9,
               label="v0.7 verdict Δ" if xi == 0 else "_nolegend_",
               edgecolor="black", linewidth=0.5)
ax.set_xticks(x)
ax.set_xticklabels(models_compare, fontsize=8, rotation=18, ha="right")
ax.axhline(0, color="black", lw=0.8, alpha=0.6)
ax.set_ylabel("logp Δ  /  verdict Δ (logit)", fontsize=9)
ax.set_title("v0.6 logp Δ vs v0.7 verdict Δ — same models", fontsize=10)
ax.legend(fontsize=8, frameon=False, loc="upper left")
ax.grid(axis="y", lw=0.3, alpha=0.4)

# ── Panel D (bottom-left): format ablation bars ─────────────────────────────
ax = axes[1, 0]
# v0.6 bare template result for 1.5B base
bare_mean = -0.0024
bare_lo = bare_hi = float("nan")
if lp_1p5b:
    vals06 = [r["delta_oracle_corrupted"] for r in lp_1p5b
              if r.get("delta_oracle_corrupted") is not None]
    bare_mean, bare_lo, bare_hi = _bootstrap_ci(vals06)

chat_vals = [r["delta_oracle_corrupted"] for r in fmt_chat
             if r.get("delta_oracle_corrupted") is not None]
int_vals  = [r["delta_oracle_corrupted"] for r in fmt_int
             if r.get("delta_oracle_corrupted") is not None]

chat_mean, chat_lo, chat_hi   = _bootstrap_ci(chat_vals)
inter_mean, inter_lo, inter_hi = _bootstrap_ci(int_vals)

templates = [
    ("bare\n(v0.6)",    bare_mean,  bare_lo,  bare_hi,  "#1f77b4"),
    ("chat",            chat_mean,  chat_lo,  chat_hi,  "#2ca02c"),
    ("interleaved",     inter_mean, inter_lo, inter_hi, "#ff7f0e"),
]
for xi, (lbl, mean, lo, hi, c) in enumerate(templates):
    _bar_with_ci(ax, xi, mean, lo, hi, color=c)
ax.set_xticks(range(len(templates)))
ax.set_xticklabels([t[0] for t in templates], fontsize=9)
ax.set_ylabel("Δ logp (oracle − corrupted)", fontsize=9)
ax.axhline(0, color="black", lw=0.8, alpha=0.6)
ax.set_title("Format ablation — Qwen-1.5B base (§4.8.3)", fontsize=10)
ax.grid(axis="y", lw=0.3, alpha=0.4)

# ── Panel E (bottom-middle): per-token surprise histogram ───────────────────
ax = axes[1, 1]
pts_valid = [r for r in pts if not r.get("skipped", False)]
lmg_vals  = [r["local_minus_global"] for r in pts_valid
             if r.get("local_minus_global") is not None]
if lmg_vals:
    arr = np.asarray(lmg_vals, dtype=float)
    ax.hist(arr, bins=20, color="#1f77b4", alpha=0.75, edgecolor="black",
            linewidth=0.5)
    ax.axvline(arr.mean(), color="#d62728", lw=1.8, ls="--",
               label=f"mean = {arr.mean():.3f}")
    ax.text(0.97, 0.95,
            f"all positive: {int(np.all(arr > 0))}  (n={arr.size})",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#2ca02c")
    ax.legend(fontsize=9, frameon=False)
else:
    ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
            ha="center", va="center", fontsize=12, color="grey")
ax.set_xlabel("local_minus_global surprise", fontsize=9)
ax.set_ylabel("count", fontsize=9)
ax.set_title("Per-token surprise: corruption window vs elsewhere (§4.8.4)",
             fontsize=10)
ax.grid(axis="y", lw=0.3, alpha=0.4)

# ── Panel F (bottom-right): per-problem v0.6 Δ vs v0.7 σ_disc (1.5B) ────────
ax = axes[1, 2]
# Build per-problem lookup from consistent_corruption_qwen_1p5b
cc_by_id = {r["id"]: r["discrimination_signal"]
            for r in cc_1p5b if r.get("discrimination_signal") is not None}
lp_by_id = {r["id"]: r["delta_oracle_corrupted"]
            for r in lp_1p5b if r.get("delta_oracle_corrupted") is not None}
common_ids = sorted(set(cc_by_id) & set(lp_by_id))
if common_ids:
    x_vals = np.array([lp_by_id[i] for i in common_ids], dtype=float)
    y_vals = np.array([cc_by_id[i] for i in common_ids], dtype=float)
    ax.scatter(x_vals, y_vals, s=22, alpha=0.75, color="#1f77b4",
               edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="grey", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(0, color="grey", lw=0.8, ls="--", alpha=0.7)
    ax.set_xlabel("v0.6 Δ logp (oracle − corrupted)", fontsize=9)
    ax.set_ylabel("v0.7 discrimination signal (logp)", fontsize=9)
    ax.set_title(
        "Per-problem dissociation — Qwen-1.5B base\n"
        "v0.6 Δ≈0 vs v0.7 σ_disc strongly positive",
        fontsize=9,
    )
    ax.grid(lw=0.3, alpha=0.4)
    # annotate counts
    n_conc = int(np.sum(y_vals > 0))
    ax.text(0.97, 0.05,
            f"{n_conc}/{len(y_vals)} σ_disc > 0",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="#2ca02c")
else:
    ax.text(0.5, 0.5, "no overlapping data", transform=ax.transAxes,
            ha="center", va="center", fontsize=11, color="grey")
    ax.set_title("Per-problem v0.6 Δ vs v0.7 σ_disc (§4.8)", fontsize=10)

fig6.tight_layout()
fig6_path = OUT / "fig6_v07_summary_panel.png"
fig6.savefig(fig6_path, dpi=150, bbox_inches="tight")
plt.close(fig6)
print(f"Wrote {fig6_path}")


# ── Figure 7: verdict Δ scaling scatter ──────────────────────────────────────
SCALE_PARAMS = {
    # key: (n_params, marker, label, family)
    "1.5B":        (1.5e9,  "o", "Qwen-1.5B base",    "base"),
    "7B":          (7.0e9,  "o", "Qwen-7B base",       "base"),
    "7B-Instruct": (7.0e9,  "s", "Qwen-7B-Instruct",   "instruct"),
    "72B":         (7.2e10, "o", "Qwen-72B base",      "base"),
    "R1-1.5B":     (1.5e9,  "^", "R1-Distill-1.5B",   "r1"),
}

fig7, ax7 = plt.subplots(figsize=(8, 5))
plotted: list[tuple] = []
for key, rows, _label in vd_models:
    vals = [r["delta_oracle_minus_corrupted"] for r in rows
            if r.get("delta_oracle_minus_corrupted") is not None]
    if not vals:
        continue
    mean = float(np.mean(vals))
    n_params, marker, label, _fam = SCALE_PARAMS[key]
    ax7.scatter(n_params, mean, marker=marker, s=80, color=MODEL_COLOR[key],
                edgecolor="black", linewidth=0.8, zorder=5)
    ax7.annotate(label, (n_params, mean),
                 textcoords="offset points", xytext=(6, 4),
                 fontsize=8, color=MODEL_COLOR[key])
    plotted.append((n_params, mean))

# connect base-model points with a thin line
base_pts = sorted(
    [(SCALE_PARAMS[k][0], float(np.mean([r["delta_oracle_minus_corrupted"]
                                          for r in rows
                                          if r.get("delta_oracle_minus_corrupted") is not None])))
     for k, rows, _ in vd_models
     if rows and SCALE_PARAMS[k][3] == "base"],
    key=lambda t: t[0]
)
if len(base_pts) > 1:
    bx, by = zip(*base_pts)
    ax7.plot(bx, by, color="grey", lw=1.0, ls="--", alpha=0.6,
             zorder=2, label="base scaling trend")

# legend for marker families
from matplotlib.lines import Line2D
legend_handles = [
    Line2D([0], [0], marker="o", color="grey", lw=0, markersize=8, label="base"),
    Line2D([0], [0], marker="s", color="grey", lw=0, markersize=8, label="instruct"),
    Line2D([0], [0], marker="^", color="grey", lw=0, markersize=8, label="R1-distill"),
]
ax7.legend(handles=legend_handles, fontsize=9, frameon=False, loc="upper left")

ax7.set_xscale("log")
ax7.set_xlabel("model parameters (log scale)", fontsize=10)
ax7.set_ylabel("mean Δ verdict logit (oracle − corrupted)", fontsize=10)
ax7.axhline(0, color="black", lw=0.8, alpha=0.6)
ax7.set_title("Verdict test: scaling — Instruct advantage visible at a glance",
              fontsize=11)
ax7.grid(lw=0.3, alpha=0.4)

fig7.tight_layout()
fig7_path = OUT / "fig7_verdict_scaling.png"
fig7.savefig(fig7_path, dpi=150, bbox_inches="tight")
plt.close(fig7)
print(f"Wrote {fig7_path}")


# ── Figure 8: per-token surprise example ─────────────────────────────────────
# The per_token_surprise JSONL contains only aggregate statistics (window vs
# elsewhere), not per-token series.  Detect this and skip gracefully.
has_series = pts and any(
    isinstance(v, list)
    for r in pts[:3]
    for v in r.values()
)

if not has_series:
    print(
        "Note: per_token_surprise_qwen_1p5b.jsonl contains only aggregate stats "
        "(no per-token series).  Falling back to an aggregate bar chart for "
        "Fig 8 instead of a token-level trace."
    )
    # Show aggregate bars for the best-case problem (largest local_minus_global)
    pts_valid2 = [r for r in pts if not r.get("skipped", False)]
    if pts_valid2:
        best = max(pts_valid2, key=lambda r: r.get("local_minus_global", -999))
        fig8, ax8 = plt.subplots(figsize=(8, 5))
        categories = [
            ("oracle\nwindow",     best["mean_surprise_oracle_at_window"]),
            ("corrupted\nwindow",  best["mean_surprise_corrupted_at_window"]),
            ("oracle\nelsewhere",  best["mean_surprise_oracle_elsewhere"]),
            ("corrupted\nelsewhere", best["mean_surprise_corrupted_elsewhere"]),
        ]
        colors8 = ["#1f77b4", "#d62728", "#aec7e8", "#ffbb78"]
        labels8 = [c[0] for c in categories]
        vals8   = [c[1] for c in categories]
        ax8.bar(range(4), vals8, color=colors8, alpha=0.85, edgecolor="black",
                linewidth=0.5)
        ax8.set_xticks(range(4))
        ax8.set_xticklabels(labels8, fontsize=9)
        ax8.set_ylabel("mean per-token surprise (cross-entropy)", fontsize=10)
        ax8.set_title(
            f"Per-token surprise breakdown — problem {best['id']} "
            f"(largest local_minus_global = {best['local_minus_global']:.3f})\n"
            f"corruption window tokens: {best['n_corruption_window_tokens']}  "
            f"(§4.8.4)",
            fontsize=10,
        )
        ax8.grid(axis="y", lw=0.3, alpha=0.4)
        fig8.tight_layout()
        fig8_path = OUT / "fig8_per_token_surprise_example.png"
        fig8.savefig(fig8_path, dpi=150, bbox_inches="tight")
        plt.close(fig8)
        print(f"Wrote {fig8_path}")
    else:
        print("Note: per_token_surprise data is empty — fig8 skipped entirely.")
else:
    # Full per-token series plot
    pts_valid3 = [r for r in pts if not r.get("skipped", False)]
    best = max(pts_valid3, key=lambda r: r.get("local_minus_global", -999))

    # Detect series key names
    oracle_key   = next((k for k in best if "oracle" in k.lower()
                          and isinstance(best[k], list)), None)
    corrupt_key  = next((k for k in best if "corrupt" in k.lower()
                          and isinstance(best[k], list)), None)

    fig8, ax8 = plt.subplots(figsize=(10, 4))
    if oracle_key and corrupt_key:
        o_ser = np.asarray(best[oracle_key], dtype=float)
        c_ser = np.asarray(best[corrupt_key], dtype=float)
        ax8.plot(o_ser, color="#1f77b4", lw=1.0, alpha=0.8, label="oracle chain")
        ax8.plot(c_ser, color="#d62728", lw=1.0, alpha=0.8, label="corrupted chain")
        fi = best.get("first_diff_idx", 0)
        li = best.get("last_diff_idx", len(o_ser) - 1)
        ax8.axvspan(fi, li, alpha=0.12, color="#d62728", label="corruption window")
        ax8.legend(fontsize=9, frameon=False)
    ax8.set_xlabel("token position", fontsize=10)
    ax8.set_ylabel("per-token surprise (cross-entropy)", fontsize=10)
    ax8.set_title(
        f"Per-token surprise — problem {best['id']} "
        f"(local−global = {best['local_minus_global']:.3f})\n"
        f"Corruption window highlighted (§4.8.4)",
        fontsize=10,
    )
    ax8.grid(lw=0.3, alpha=0.4)
    fig8.tight_layout()
    fig8_path = OUT / "fig8_per_token_surprise_example.png"
    fig8.savefig(fig8_path, dpi=150, bbox_inches="tight")
    plt.close(fig8)
    print(f"Wrote {fig8_path}")

print("done")

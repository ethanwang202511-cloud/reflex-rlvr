"""Local re-analysis of all corrective-probe JSONL files.

For each probe (consistent_corruption, verdict, per_token_surprise) and
each model variant: compute paired-bootstrap 95% CIs and exact effect
sizes, write a per-probe CSV table.

Produces:
  results/pilot/premise/analysis/consistent_corruption_summary.csv
  results/pilot/premise/analysis/verdict_summary.csv
  results/pilot/premise/analysis/per_token_surprise_summary.csv
  results/pilot/premise/analysis/coverage_matrix.csv
"""
from __future__ import annotations
import csv
import json
import math
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREMISE = ROOT / "results" / "pilot" / "premise"
OUT = PREMISE / "analysis"
OUT.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(31415)
N_BOOT = 1000

def paired_bootstrap_ci(values, agg=np.mean, alpha=0.05, n_boot=N_BOOT):
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return (float("nan"), float("nan"), float("nan"))
    boot = np.empty(n_boot)
    n = len(arr)
    for b in range(n_boot):
        idx = RNG.integers(0, n, n)
        boot[b] = agg(arr[idx])
    lo = float(np.percentile(boot, 100 * alpha / 2))
    hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return (float(agg(arr)), lo, hi)

def cohens_d_paired(x, y):
    """Paired Cohen's d for two same-length arrays."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    d = x - y
    d = d[~np.isnan(d)]
    if len(d) == 0:
        return float("nan")
    sd = np.std(d, ddof=1)
    if sd == 0:
        return float("nan")
    return float(np.mean(d) / sd)

def cliffs_delta(x, y):
    """Cliff's delta for paired arrays."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]; y = y[mask]
    if len(x) == 0:
        return float("nan")
    pos = np.sum(x > y); neg = np.sum(x < y)
    return float((pos - neg) / len(x))

def load_jsonl(p):
    rows = []
    with open(p) as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            rows.append(json.loads(ln))
    return rows

def slug_to_model(slug):
    # qwen_1p5b → Qwen2.5-1.5B (display name)
    m = {
        "qwen_1p5b": "Qwen2.5-1.5B (base)",
        "qwen_1p5b_instruct": "Qwen2.5-1.5B-Instruct",
        "qwen_7b": "Qwen2.5-7B (base)",
        "qwen_7b_instruct": "Qwen2.5-7B-Instruct",
        "qwen_72b": "Qwen2.5-72B (base)",
        "r1_qwen_1p5b": "DeepSeek-R1-Distill-Qwen-1.5B",
        "r1_qwen_7b": "DeepSeek-R1-Distill-Qwen-7B",
        "r1_llama_8b": "DeepSeek-R1-Distill-Llama-8B",
        "llama_3p1_8b": "Llama-3.1-8B (base)",
        "mistral_7b": "Mistral-7B-v0.3 (base)",
        "gemma_9b": "Gemma-2-9B (base)",
        "1p5b": "Qwen2.5-1.5B (base)",
        "7b": "Qwen2.5-7B (base)",
        "1p5b_instruct": "Qwen2.5-1.5B-Instruct",
        "7b_instruct": "Qwen2.5-7B-Instruct",
    }
    return m.get(slug, slug)

# ------------------------------------------------------------------
# CONSISTENT CORRUPTION
# ------------------------------------------------------------------
cc_files = sorted(PREMISE.glob("consistent_corruption_*.jsonl"))
cc_rows = []
for fp in cc_files:
    slug = fp.stem.replace("consistent_corruption_", "")
    rows = load_jsonl(fp)
    n = len(rows)
    disc = [r.get("discrimination_signal") for r in rows]
    do = [r.get("delta_disc_oracle") for r in rows]
    dc = [r.get("delta_disc_corrupted") for r in rows]
    disc_clean = [v for v in disc if v is not None and not (isinstance(v, float) and math.isnan(v))]
    do_clean = [v for v in do if v is not None and not (isinstance(v, float) and math.isnan(v))]
    dc_clean = [v for v in dc if v is not None and not (isinstance(v, float) and math.isnan(v))]
    mean_disc, lo_disc, hi_disc = paired_bootstrap_ci(disc_clean)
    mean_do, lo_do, hi_do = paired_bootstrap_ci(do_clean)
    mean_dc, lo_dc, hi_dc = paired_bootstrap_ci(dc_clean)
    pct_pos = float(np.mean(np.asarray(disc_clean) > 0)) if disc_clean else float("nan")
    # cohens d for delta_disc_oracle vs delta_disc_corrupted
    d_paired = cohens_d_paired(do_clean, dc_clean) if len(do_clean) == len(dc_clean) else float("nan")
    cliff = cliffs_delta(do_clean, dc_clean) if len(do_clean) == len(dc_clean) else float("nan")
    cc_rows.append({
        "model": slug_to_model(slug),
        "slug": slug,
        "n_problems": n,
        "mean_disc_signal": round(mean_disc, 4),
        "disc_signal_ci_lo": round(lo_disc, 4),
        "disc_signal_ci_hi": round(hi_disc, 4),
        "pct_oracle_above": round(pct_pos, 3),
        "mean_delta_oracle": round(mean_do, 4),
        "mean_delta_corrupted": round(mean_dc, 4),
        "cohens_d_paired": round(d_paired, 4),
        "cliffs_delta": round(cliff, 4),
        "gate_passed": bool(pct_pos > 0.6 and mean_disc > 0),
    })
print(f"consistent_corruption: {len(cc_files)} models analyzed")

with open(OUT / "consistent_corruption_summary.csv", "w") as f:
    w = csv.DictWriter(f, fieldnames=list(cc_rows[0].keys()))
    w.writeheader()
    w.writerows(cc_rows)

# ------------------------------------------------------------------
# VERDICT
# ------------------------------------------------------------------
v_files = sorted(PREMISE.glob("verdict_*.jsonl"))
v_rows = []
for fp in v_files:
    slug = fp.stem.replace("verdict_", "")
    rows = load_jsonl(fp)
    delta = [r.get("delta_oracle_minus_corrupted") for r in rows]
    delta_clean = [v for v in delta if v is not None and not (isinstance(v, float) and math.isnan(v))]
    lo_o = [r.get("logit_oracle") for r in rows if r.get("logit_oracle") is not None]
    lo_c = [r.get("logit_corrupted") for r in rows if r.get("logit_corrupted") is not None]
    lo_s = [r.get("logit_shuffled") for r in rows if r.get("logit_shuffled") is not None]
    mean_d, lo_d, hi_d = paired_bootstrap_ci(delta_clean)
    pct_pos = float(np.mean(np.asarray(delta_clean) > 0)) if delta_clean else float("nan")
    coh = cohens_d_paired([r.get("logit_oracle") for r in rows], [r.get("logit_corrupted") for r in rows])
    v_rows.append({
        "model": slug_to_model(slug),
        "slug": slug,
        "n_problems": len(rows),
        "mean_logit_oracle": round(float(np.mean(lo_o)), 4) if lo_o else float("nan"),
        "mean_logit_corrupted": round(float(np.mean(lo_c)), 4) if lo_c else float("nan"),
        "mean_logit_shuffled": round(float(np.mean(lo_s)), 4) if lo_s else float("nan"),
        "mean_delta_verdict": round(mean_d, 4),
        "delta_ci_lo": round(lo_d, 4),
        "delta_ci_hi": round(hi_d, 4),
        "pct_oracle_above_corrupted": round(pct_pos, 3),
        "cohens_d_paired": round(coh, 4),
        "gate_passed": bool(pct_pos > 0.6 and mean_d > 0),
    })
print(f"verdict: {len(v_files)} models analyzed")

with open(OUT / "verdict_summary.csv", "w") as f:
    w = csv.DictWriter(f, fieldnames=list(v_rows[0].keys()))
    w.writeheader()
    w.writerows(v_rows)

# ------------------------------------------------------------------
# PER-TOKEN SURPRISE
# ------------------------------------------------------------------
s_files = sorted(PREMISE.glob("per_token_surprise_*.jsonl"))
s_rows = []
for fp in s_files:
    slug = fp.stem.replace("per_token_surprise_", "")
    rows = load_jsonl(fp)
    dw = [r.get("delta_at_window") for r in rows]
    de = [r.get("delta_elsewhere") for r in rows]
    dw_clean = [v for v in dw if v is not None and not (isinstance(v, float) and math.isnan(v))]
    de_clean = [v for v in de if v is not None and not (isinstance(v, float) and math.isnan(v))]
    lmg = [w - e for w, e in zip(dw_clean, de_clean[:len(dw_clean)])] if dw_clean else []
    mean_lmg, lo_lmg, hi_lmg = paired_bootstrap_ci(lmg) if lmg else (float("nan"),)*3
    mean_w, lo_w, hi_w = paired_bootstrap_ci(dw_clean) if dw_clean else (float("nan"),)*3
    mean_e, lo_e, hi_e = paired_bootstrap_ci(de_clean) if de_clean else (float("nan"),)*3
    pct_pos = float(np.mean(np.asarray(lmg) > 0)) if lmg else float("nan")
    coh = cohens_d_paired(dw_clean, de_clean[:len(dw_clean)]) if dw_clean else float("nan")
    s_rows.append({
        "model": slug_to_model(slug),
        "slug": slug,
        "n_problems": len(rows),
        "mean_delta_at_window": round(mean_w, 4),
        "window_ci_lo": round(lo_w, 4),
        "window_ci_hi": round(hi_w, 4),
        "mean_delta_elsewhere": round(mean_e, 4),
        "elsewhere_ci_lo": round(lo_e, 4),
        "elsewhere_ci_hi": round(hi_e, 4),
        "mean_local_minus_global": round(mean_lmg, 4),
        "lmg_ci_lo": round(lo_lmg, 4),
        "lmg_ci_hi": round(hi_lmg, 4),
        "pct_positive_local_minus_global": round(pct_pos, 3),
        "cohens_d_paired": round(coh, 4),
        "gate_passed": bool(pct_pos > 0.6 and mean_lmg > 0),
    })
print(f"per_token_surprise: {len(s_files)} models analyzed")

with open(OUT / "per_token_surprise_summary.csv", "w") as f:
    w = csv.DictWriter(f, fieldnames=list(s_rows[0].keys()))
    w.writeheader()
    w.writerows(s_rows)

# ------------------------------------------------------------------
# COVERAGE MATRIX
# ------------------------------------------------------------------
all_models = set()
for rows in [cc_rows, v_rows, s_rows]:
    for r in rows:
        all_models.add(r["model"])

cov = {}
for r in cc_rows: cov.setdefault(r["model"], {})["consistent_corruption"] = "PASS" if r["gate_passed"] else "FAIL"
for r in v_rows: cov.setdefault(r["model"], {})["verdict"] = "PASS" if r["gate_passed"] else "FAIL"
for r in s_rows: cov.setdefault(r["model"], {})["per_token_surprise"] = "PASS" if r["gate_passed"] else "FAIL"

with open(OUT / "coverage_matrix.csv", "w") as f:
    w = csv.DictWriter(f, fieldnames=["model", "consistent_corruption", "verdict", "per_token_surprise"])
    w.writeheader()
    for m in sorted(all_models):
        w.writerow({"model": m, **{p: cov.get(m, {}).get(p, "—") for p in ["consistent_corruption","verdict","per_token_surprise"]}})

print(f"\nWrote 4 CSVs to {OUT}")
print("Summary of consistent_corruption gates:")
for r in cc_rows:
    print(f"  {r['model']:<35s}  disc_signal={r['mean_disc_signal']:.3f} [{r['disc_signal_ci_lo']:.3f},{r['disc_signal_ci_hi']:.3f}]  pct={r['pct_oracle_above']:.3f}  d={r['cohens_d_paired']:.3f}  GATE={'PASS' if r['gate_passed'] else 'FAIL'}")
print("\nVerdict gates:")
for r in v_rows:
    print(f"  {r['model']:<35s}  delta={r['mean_delta_verdict']:+.4f} [{r['delta_ci_lo']:+.4f},{r['delta_ci_hi']:+.4f}]  pct={r['pct_oracle_above_corrupted']:.3f}  d={r['cohens_d_paired']:+.3f}  GATE={'PASS' if r['gate_passed'] else 'FAIL'}")
print("\nPer-token surprise gates:")
for r in s_rows:
    print(f"  {r['model']:<35s}  L-G={r['mean_local_minus_global']:+.4f} [{r['lmg_ci_lo']:+.4f},{r['lmg_ci_hi']:+.4f}]  pct={r['pct_positive_local_minus_global']:.3f}  GATE={'PASS' if r['gate_passed'] else 'FAIL'}")

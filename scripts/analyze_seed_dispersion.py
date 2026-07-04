"""Seed dispersion analysis for the J4 multi-seed runs.

For each (model, probe) cell that has multiple seeds, aggregate the
headline statistic into mean ± std across seeds, with 4-seed coverage
(1337 + 37 + 31415 + 271828).

Writes results/pilot/premise/analysis/seed_dispersion.csv.
"""
from __future__ import annotations
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, stdev

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PREMISE = ROOT / "results" / "pilot" / "premise"
OUT = PREMISE / "analysis" / "seed_dispersion.csv"

SEED_SLUG_RE = re.compile(r"_seed(\d+)$")

def load_jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

def base_stat(rows, key):
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    vals = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    return float(np.mean(vals)) if vals else float("nan")

def collect(probe, headline_key):
    """Group files by model; assign 1337 to baseline (no _seedN suffix), and seeds to others."""
    by_model = {}
    for fp in sorted(PREMISE.glob(f"{probe}_*.jsonl")):
        slug = fp.stem.replace(f"{probe}_", "")
        m = SEED_SLUG_RE.search(slug)
        if m:
            seed = int(m.group(1))
            model_slug = SEED_SLUG_RE.sub("", slug)
        else:
            seed = 1337  # default seed for single-seed runs
            model_slug = slug
        rows = load_jsonl(fp)
        val = base_stat(rows, headline_key)
        by_model.setdefault(model_slug, {})[seed] = (val, len(rows))
    return by_model

cc = collect("consistent_corruption", "discrimination_signal")
v  = collect("verdict", "delta_oracle_minus_corrupted")

rows_out = []

def summarize(probe_name, headline_label, by_model):
    for model_slug, seed_map in by_model.items():
        if len(seed_map) < 2:
            continue  # need at least 2 seeds to compute dispersion
        seeds = sorted(seed_map.keys())
        vals = [seed_map[s][0] for s in seeds]
        ns = [seed_map[s][1] for s in seeds]
        rows_out.append({
            "probe": probe_name,
            "headline": headline_label,
            "model_slug": model_slug,
            "n_seeds": len(seeds),
            "seeds": ",".join(str(s) for s in seeds),
            "values": ",".join(f"{v:.4f}" for v in vals),
            "mean": round(mean(vals), 4),
            "std": round(stdev(vals), 4) if len(vals) >= 2 else float("nan"),
            "min": round(min(vals), 4),
            "max": round(max(vals), 4),
            "cv_pct": round(100 * stdev(vals) / abs(mean(vals)), 2) if (len(vals) >= 2 and mean(vals) != 0) else float("nan"),
        })

summarize("consistent_corruption", "sigma_disc", cc)
summarize("verdict",                "delta_verdict", v)

with open(OUT, "w") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    w.writeheader()
    w.writerows(rows_out)

print(f"Wrote {OUT}")
print()
for r in rows_out:
    print(f"  {r['probe']:25s} {r['model_slug']:25s}  n_seeds={r['n_seeds']}  values=[{r['values']}]  mean±std={r['mean']:+.4f}±{r['std']:.4f}  CV={r['cv_pct']}%")

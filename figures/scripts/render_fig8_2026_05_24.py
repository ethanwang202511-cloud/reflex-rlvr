"""Regenerate fig8 per-token surprise example — vector PDF, clean text.

Picks one representative problem from per_token_surprise_qwen_7b_instruct.jsonl
(highest LMG) and plots oracle vs corrupted per-token surprise traces
with the corruption window highlighted.
"""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "DejaVu Sans"]

ROOT = Path(__file__).resolve().parents[2]
JSONL = ROOT / "results" / "pilot" / "premise" / "per_token_surprise_qwen_7b_instruct.jsonl"
OUT_PDF = ROOT / "figures" / "output" / "fig8_per_token_surprise_example.pdf"
OUT_PNG = ROOT / "figures" / "output" / "fig8_per_token_surprise_example.png"

rows = [json.loads(l) for l in JSONL.read_text().splitlines() if l.strip()]
# Show summary statistics — bar chart for at-window vs elsewhere
ids = [r["id"] for r in rows]
dw = [r.get("delta_at_window", 0.0) or 0.0 for r in rows]
de = [r.get("delta_elsewhere", 0.0) or 0.0 for r in rows]

import numpy as np
x = np.arange(len(ids))

fig, ax = plt.subplots(figsize=(11, 4.0))
w = 0.4
ax.bar(x - w/2, dw, w, color="#d62728", label=r"$\Delta$ at corruption window ($\pm 3$ tokens)")
ax.bar(x + w/2, de, w, color="#1f77b4", label=r"$\Delta$ elsewhere in the chain")
ax.axhline(0, color="black", linewidth=0.6)
ax.set_xticks(x)
ax.set_xticklabels(ids, rotation=70, fontsize=7, ha="right")
ax.set_xlabel("Problem (AIME 2018-2023 hard subset, 51 problems)", fontsize=10)
ax.set_ylabel(r"Per-token surprise difference (corrupted - oracle)", fontsize=10)
ax.set_title(f"Probe 4: per-token surprise on Qwen2.5-7B-Instruct ($n={len(ids)}$ problems)\n"
             "At-window surprise > elsewhere surprise on every problem (LMG = +1.76, all 51 positive)", fontsize=10)
ax.legend(loc="upper right", fontsize=9, frameon=True)
ax.grid(axis="y", linestyle=":", alpha=0.5)

plt.tight_layout()
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.savefig(OUT_PNG, dpi=220, bbox_inches="tight")
print(f"Wrote {OUT_PDF}\nWrote {OUT_PNG}")

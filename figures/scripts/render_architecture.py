"""Render the self-teacher RLVR architecture diagram.

Shows the verifier (world feedback, clean and binary) and the base's
internal discriminator (self-generated feedback, measurement-gated and
real-valued) as two feedback channels that close the loop back into
the base model. Annotates the four measurement pathways on the
discriminator side.

No external data dependency. Pure matplotlib drawing.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = REPO_ROOT / "figures" / "output"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
})


def box(ax, xy, w, h, label, facecolor, edgecolor, fontsize=9, fontweight="normal"):
    rect = mpatches.FancyBboxPatch(
        (xy[0] - w / 2, xy[1] - h / 2),
        w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.2,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(rect)
    ax.text(xy[0], xy[1], label, ha="center", va="center",
            fontsize=fontsize, fontweight=fontweight)


def arrow(ax, p0, p1, color="#444", lw=1.4, style="-", connectionstyle=None):
    if connectionstyle is None:
        ax.annotate("", xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle="-|>",
                                    color=color, lw=lw,
                                    linestyle=style,
                                    mutation_scale=12))
    else:
        ax.annotate("", xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle="-|>",
                                    color=color, lw=lw,
                                    linestyle=style,
                                    mutation_scale=12,
                                    connectionstyle=connectionstyle))


def main():
    fig, ax = plt.subplots(figsize=(11.0, 4.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_aspect("auto")
    ax.set_xticks([])
    ax.set_yticks([])

    base_color = "#dbe7f6"
    base_edge = "#3a6ea5"
    verifier_color = "#d8efd8"
    verifier_edge = "#2e7d32"
    disc_color = "#fde9c8"
    disc_edge = "#c97500"
    reward_color = "#eeeeee"
    reward_edge = "#555"

    # Base LLM at top
    box(ax, (5, 5.2), 2.6, 0.7, "Base LLM (generator)",
        base_color, base_edge, fontweight="bold")

    # Chain output
    ax.text(5, 4.55, r"chain $y$ for problem $x$",
            ha="center", va="center", fontsize=8, style="italic")
    arrow(ax, (5, 4.85), (5, 4.7))

    # Verifier (left)
    box(ax, (2.4, 3.2), 2.6, 1.05,
        "Verifier\n(SymPy, Lean 4,\ncode executor)",
        verifier_color, verifier_edge, fontweight="bold")
    ax.text(2.4, 2.42, "world feedback",
            ha="center", va="center", fontsize=8, style="italic", color="#2e7d32")
    ax.text(2.4, 2.15, r"$r \in \{0, 1\}$, clean, terminal",
            ha="center", va="center", fontsize=8, color="#444")

    # Discriminator (right)
    box(ax, (7.6, 3.2), 2.6, 1.05,
        "Base discriminator\n(four measurement\npathways)",
        disc_color, disc_edge, fontweight="bold")
    ax.text(7.6, 2.42, "self-generated feedback",
            ha="center", va="center", fontsize=8, style="italic", color="#c97500")
    ax.text(7.6, 2.15, r"$\Delta \in \mathbb{R}$, measurement-gated",
            ha="center", va="center", fontsize=8, color="#444")

    # Chain → verifier and discriminator
    arrow(ax, (4.3, 4.45), (3.0, 3.7))
    arrow(ax, (5.7, 4.45), (7.0, 3.7))

    # Probe pathway labels under discriminator
    probes = "visible-answer logprob  $\\bullet$  verdict YES/NO  $\\bullet$  per-token surprise  $\\bullet$  answer-relabeled"
    ax.text(7.6, 1.75, probes, ha="center", va="center", fontsize=9, color="#666")

    # Reward aggregator
    box(ax, (5, 1.0), 2.6, 0.6, "reward aggregator",
        reward_color, reward_edge)

    # Arrows to aggregator
    arrow(ax, (2.9, 2.7), (4.1, 1.2))
    arrow(ax, (7.1, 2.7), (5.9, 1.2))

    # Aggregator → base (loop on the right)
    arrow(ax, (6.3, 1.0), (6.4, 5.0), color="#555",
          connectionstyle="arc3,rad=0.4", style="--")
    ax.text(8.6, 3.0, "policy\nupdate",
            ha="left", va="center", fontsize=7.5, style="italic", color="#555")

    # Pipeline annotations (bottom)
    ax.text(2.4, 0.35,
            "R1-Zero-style:\nverifier only",
            ha="center", va="center", fontsize=7.5, color="#2e7d32",
            fontweight="bold")
    ax.text(7.6, 0.35,
            "LDPT / self-teacher:\nboth channels",
            ha="center", va="center", fontsize=7.5, color="#c97500",
            fontweight="bold")

    fig.tight_layout()
    out_path = OUT / "fig_architecture.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()

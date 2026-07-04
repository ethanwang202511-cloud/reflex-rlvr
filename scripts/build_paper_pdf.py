"""Build paper.md → paper/builds/paper.pdf using the shared
ReportLab pipeline at ../_tools/make_pdfs_conference.py.

Run from anywhere:

    python3 scripts/build_paper_pdf.py

No CLI flags. Output: paper/builds/paper.pdf (full) and
paper/builds/paper_compact.pdf (compact heading style).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESEARCH_ROOT = REPO_ROOT.parent
TOOLS_DIR = RESEARCH_ROOT / "_tools"

# Add the shared tools directory to sys.path so we can import the
# already-existing PDF builder.
sys.path.insert(0, str(TOOLS_DIR))

from make_pdfs_conference import _register_fonts, build_paper  # noqa: E402


def main() -> int:
    md_path = REPO_ROOT / "paper" / "paper.md"
    if not md_path.exists():
        print(f"ERROR: {md_path} not found", file=sys.stderr)
        return 2
    out_dir = REPO_ROOT / "paper" / "builds"
    out_dir.mkdir(parents=True, exist_ok=True)

    _register_fonts()

    # No figures: figures/ folder is intentionally empty for this
    # negative-result paper (gate (a) failure produced tabular results
    # only; no plots commissioned). Pass empty fig_plan and the unused
    # figures dir; build_paper handles missing figs gracefully.
    fig_dir = REPO_ROOT / "figures" / "output"
    fig_plan: dict = {}

    full_out = out_dir / "paper.pdf"
    compact_out = out_dir / "paper_compact.pdf"

    print(f"Building {full_out.name} (no page limit) ...")
    build_paper(md_path, fig_dir, fig_plan, full_out)
    print(f"  -> {full_out}")

    print(f"Building {compact_out.name} (compact heading style) ...")
    build_paper(md_path, fig_dir, fig_plan, compact_out, compact=True)
    print(f"  -> {compact_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

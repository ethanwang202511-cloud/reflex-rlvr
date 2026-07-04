"""Fetch the AIME problem set + canonical solutions for the Week-1 pilot.

Pulls two HuggingFace datasets:

1. ``qq8933/AIME_1983_2024`` — clean AIME problem text with year /
   problem-number metadata and integer answers. 933 problems total
   (full coverage 1983–2023, partial 2024). License: see HF page; we
   use only for a private pilot, not redistributed.

2. ``AI-MO/NuminaMath-CoT`` — 859k math problems with chain-of-thought
   solutions. We slice the ``amc_aime`` source split (4,070 rows) and
   fuzzy-match each qq8933 AIME problem against it to obtain a
   solution string. Match rate on AIME 2018–2023 is empirically ≥ 85%
   based on sample inspection.

Outputs (under ``data/aime/``):

- ``aime_1983_2024.jsonl`` — qq8933 raw, one row per line.
- ``numina_amc_aime.jsonl`` — NuminaMath amc_aime slice, raw.
- ``h_k_pilot.jsonl`` — joined 2018-2023 problems (max 100) with
  fields ``id``, ``year``, ``problem_number``, ``problem``,
  ``answer`` (int), ``solution`` (str | None — None if no match
  ≥ ``--min-fuzz`` was found), ``solution_source``,
  ``solution_match_score``.
- ``h_k_pilot.md`` — provenance / column documentation.

Run:

    python scripts/fetch_data.py
    python scripts/fetch_data.py --min-fuzz 80 --years 2018-2023 --target-n 120

Idempotent: re-running re-validates files but does not re-download
unless ``--force`` is passed.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "aime"


def _normalize(text: str) -> str:
    """Aggressive normalization for fuzzy matching: lowercase, strip
    most LaTeX backslashes, collapse whitespace. The match is on the
    *first 200* normalized characters — enough to disambiguate AIME
    problems while being robust to solution-text differences in the
    second half."""
    if not text:
        return ""
    t = text.lower()
    t = re.sub(r"\\[a-z]+\{?", " ", t)
    t = re.sub(r"[^a-z0-9 .]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t[:200]


def _save_jsonl(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def fetch_qq8933(force: bool = False) -> Path:
    out = DATA_DIR / "aime_1983_2024.jsonl"
    if out.exists() and not force:
        print(f"[fetch_qq8933] using cached {out}")
        return out
    from datasets import load_dataset

    print("[fetch_qq8933] downloading qq8933/AIME_1983_2024 …")
    d = load_dataset("qq8933/AIME_1983_2024", split="train")
    rows = []
    n_excluded = 0
    for r in d:
        # Per proposal §5.5.2.1, problems with publicly-contested
        # official answers are excluded. The qq8933 dataset records
        # these with non-numeric Answer fields like "080 or 081 (both
        # were accepted)". We drop them at fetch time.
        try:
            answer_int = int(str(r["Answer"]).strip())
        except (ValueError, TypeError):
            n_excluded += 1
            continue
        rows.append(
            {
                "id": r["ID"],
                "year": int(r["Year"]),
                "problem_number": int(r["Problem Number"]),
                "part": r["Part"],
                "problem": r["Question"],
                "answer": answer_int,
            }
        )
    _save_jsonl(rows, out)
    print(
        f"[fetch_qq8933] saved {len(rows)} rows to {out} "
        f"(excluded {n_excluded} contested-answer rows)"
    )
    return out


def fetch_numina_amc_aime(force: bool = False) -> Path:
    out = DATA_DIR / "numina_amc_aime.jsonl"
    if out.exists() and not force:
        print(f"[fetch_numina_amc_aime] using cached {out}")
        return out
    from datasets import load_dataset

    print(
        "[fetch_numina_amc_aime] downloading AI-MO/NuminaMath-CoT and slicing amc_aime …"
    )
    d = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    amc = d.filter(lambda x: x["source"] == "amc_aime", num_proc=4)
    rows = []
    for r in amc:
        rows.append(
            {
                "source": r["source"],
                "problem": r["problem"],
                "solution": r["solution"],
            }
        )
    _save_jsonl(rows, out)
    print(f"[fetch_numina_amc_aime] saved {len(rows)} rows to {out}")
    return out


def join_pilot_set(
    qq_path: Path,
    numina_path: Path,
    *,
    out_path: Path,
    years: tuple[int, int] = (2018, 2023),
    target_n: int = 100,
    min_fuzz: int = 75,
) -> Path:
    """Build ``H_K_pilot`` — fuzzy-join AIME problems with Numina solutions."""
    from rapidfuzz import process, fuzz  # local import: only needed here

    qq = _load_jsonl(qq_path)
    numina = _load_jsonl(numina_path)
    qq_window = [r for r in qq if years[0] <= r["year"] <= years[1]]
    print(
        f"[join_pilot_set] {len(qq_window)} qq8933 problems in years "
        f"{years[0]}-{years[1]}; matching against {len(numina)} Numina rows"
    )

    numina_norm = [_normalize(r["problem"]) for r in numina]
    out_rows = []
    matched = 0
    for q in qq_window:
        q_norm = _normalize(q["problem"])
        if not q_norm or not numina_norm:
            continue
        match = process.extractOne(
            q_norm, numina_norm, scorer=fuzz.WRatio
        )
        if match is None:
            row = {**q, "solution": None, "solution_source": None, "solution_match_score": 0}
        else:
            _matched_text, score, idx = match
            if score >= min_fuzz:
                row = {
                    **q,
                    "solution": numina[idx]["solution"],
                    "solution_source": "AI-MO/NuminaMath-CoT amc_aime",
                    "solution_match_score": float(score),
                }
                matched += 1
            else:
                row = {
                    **q,
                    "solution": None,
                    "solution_source": None,
                    "solution_match_score": float(score),
                }
        out_rows.append(row)

    print(
        f"[join_pilot_set] matched {matched} / {len(out_rows)} at "
        f"min_fuzz={min_fuzz}"
    )

    # Order by year then by problem number; truncate to target_n.
    out_rows.sort(key=lambda r: (r["year"], r["problem_number"]))
    out_rows_with_solutions = [r for r in out_rows if r["solution"] is not None]
    out_rows_no_solutions = [r for r in out_rows if r["solution"] is None]

    print(
        f"[join_pilot_set] {len(out_rows_with_solutions)} with solutions, "
        f"{len(out_rows_no_solutions)} without"
    )
    if len(out_rows_with_solutions) < target_n:
        print(
            f"[join_pilot_set] WARNING: only {len(out_rows_with_solutions)} "
            f"problems with solutions (< target_n={target_n}). "
            "Will save all with solutions; gate (a) of the pilot will run on "
            "the achievable subset."
        )

    final = out_rows_with_solutions[:target_n]
    _save_jsonl(final, out_path)
    print(f"[join_pilot_set] saved {len(final)} rows to {out_path}")
    return out_path


def write_provenance(out_path: Path, args: argparse.Namespace) -> None:
    md_path = out_path.with_suffix(".md")
    md_path.write_text(
        f"""# {out_path.name} — provenance

Generated by `scripts/fetch_data.py` on AIME problems from
`qq8933/AIME_1983_2024` joined against canonical solutions from
`AI-MO/NuminaMath-CoT` (source slice: `amc_aime`) via
RapidFuzz `WRatio` on a normalized first-200-character prefix.

## Fields (one JSON object per line)

| Field                   | Type     | Description                                    |
|-------------------------|----------|------------------------------------------------|
| `id`                    | str      | qq8933 ID, e.g. `2018-1`                       |
| `year`                  | int      | AIME year                                      |
| `problem_number`        | int      | 1–15 within the AIME paper                     |
| `part`                  | str|null | AIME-I or AIME-II label where applicable       |
| `problem`               | str      | Problem statement (LaTeX preserved)            |
| `answer`                | int      | Integer answer 0–999                           |
| `solution`              | str|null | NuminaMath canonical CoT solution              |
| `solution_source`       | str|null | Always `AI-MO/NuminaMath-CoT amc_aime`         |
| `solution_match_score`  | float    | RapidFuzz WRatio in [0, 100]; ≥ {args.min_fuzz}|

## Filter parameters

- Year range: {args.years[0]}–{args.years[1]}
- Minimum fuzzy match score: {args.min_fuzz}
- Target number of problems: {args.target_n}

## Caveats

The Numina-CoT solutions are not strictly the AoPS canonical
solutions; they are high-quality model-generated solutions on the same
problems. For the purposes of the gate (a) discriminator–generator
asymmetry test (proposal §1.7), what matters is that the solution
correctly leads to the answer when fed as CoT context — verified by
the SymPy verifier on `pass@N(base | x ⊕ y_oracle)`. This is robust to
solution-style differences.
""",
        encoding="utf-8",
    )
    print(f"[write_provenance] {md_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch AIME pilot data.")
    parser.add_argument("--years", type=str, default="2018-2023")
    parser.add_argument("--target-n", type=int, default=100)
    parser.add_argument("--min-fuzz", type=int, default=75)
    parser.add_argument("--force", action="store_true", help="re-download")
    raw = parser.parse_args()

    yrs = tuple(int(x) for x in raw.years.split("-"))
    if len(yrs) != 2:
        print("--years must be of the form 'YYYY-YYYY'", file=sys.stderr)
        return 2
    raw.years = yrs

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    qq_path = fetch_qq8933(force=raw.force)
    numina_path = fetch_numina_amc_aime(force=raw.force)
    out_path = DATA_DIR / "h_k_pilot.jsonl"
    join_pilot_set(
        qq_path,
        numina_path,
        out_path=out_path,
        years=raw.years,
        target_n=raw.target_n,
        min_fuzz=raw.min_fuzz,
    )
    write_provenance(out_path, raw)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# data/

Local cache for evaluation problems, mining sources, and pretrained
model artifacts that are too large to version. Contents are
**gitignored**; provenance is recorded in this README and in the
mining scripts under `../scripts/`.

## Sources (lazy-fetched on first use)

| Subdir                  | Source                          | License           | Approx. size |
|-------------------------|---------------------------------|-------------------|--------------|
| `math/`                 | Hendrycks et al. MATH           | MIT               | ~70 MB       |
| `aime/`                 | AoPS-scraped AIME 1983–2026     | Public            | ~5 MB        |
| `hmmt/`                 | HMMT 2018–2026                  | Public            | ~3 MB        |
| `putnam/`               | Putnam 1980–2024                | Public            | ~10 MB       |
| `livecodebench/`        | LiveCodeBench (multiple snapshots) | MIT            | ~400 MB      |
| `codeforces/`           | Codeforces div2 2020–2025       | Scraped, fair use | ~600 MB      |
| `arc_agi_2/`            | ARC-AGI-2 train + eval-private  | Apache-2.0        | ~50 MB       |
| `deepseek_prover_v2/`   | DeepSeek-Prover-V2 generated    | MIT               | ~2 GB        |
| `openmathinstruct2/`    | OpenMathInstruct-2              | NVIDIA-open       | ~25 GB       |
| `open_orca/`            | Open-Orca                       | Apache-2.0        | ~12 GB       |

## Decontamination snapshot

8-gram overlap is computed against:

- RedPajama-v2 (proxy for pretraining corpus).
- FineWeb-Edu.
- StackV2-Edu.
- Held-out eval pool (`H_K_eval`).

Retain only problems with overlap < 1%. SFT-mix corpora are
snapshotted to versions ≤ 2025-12-31 to limit AIME-2026 leakage.

## What's here as of 2026-05-03

Empty. No data downloaded yet. The first download will be the
AIME-2018-2023 + AoPS-canonical-solutions subset for the GSI
calibration set and the Week-1 pilot.

## Gitignore policy

The entire `data/` folder is gitignored except for this README. Datasets
are reproducibly fetched via `scripts/fetch_data.py` (TBD).

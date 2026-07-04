"""pass@k estimator — the headline metric.

We use the unbiased Chen et al. (HumanEval) estimator:

    pass@k = E_x [ 1 - C(n - c, k) / C(n, k) ]    if n - c >= k
           = 1                                    if n - c <  k

where for problem ``x``: ``n`` is the number of samples drawn and
``c`` is the number of correct samples. This is *unbiased* —
distinct from the naive ``1 - (1 - p̂)^k`` plug-in estimator we use
inline elsewhere (which is biased toward 1 at small n).

For mining (where we drop problems with c=0 over n=1024) the
distinction does not matter; for the headline `Δ pass@k` it does.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterable, Sequence

import numpy as np


def _comb(n: int, k: int) -> int:
    if k < 0 or k > n:
        return 0
    return math.comb(n, k)


def pass_at_k_estimator(n: int, c: int, k: int) -> float:
    """Unbiased single-problem pass@k from n samples with c correct.

    Returns 0.0 when k > n (caller error / extrapolation guard).
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if c < 0 or c > n:
        raise ValueError(f"c must be in [0, n]; got c={c}, n={n}")
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    if k > n:
        return 0.0  # cannot estimate by this method
    if n - c < k:
        return 1.0
    return 1.0 - _comb(n - c, k) / _comb(n, k)


def pass_at_k_per_problem(
    correctness_records: Sequence[Sequence[int]],
    k: int,
) -> list[float]:
    """Apply :func:`pass_at_k_estimator` per problem.

    ``correctness_records[i]`` is a list of 0/1 ints — the per-sample
    correctness for problem ``i``.
    """
    out = []
    for record in correctness_records:
        n = len(record)
        c = int(sum(record))
        out.append(pass_at_k_estimator(n, c, k))
    return out


def mean_pass_at_k(
    correctness_records: Sequence[Sequence[int]],
    k: int,
) -> float:
    """Mean over problems of the per-problem pass@k estimator."""
    per_problem = pass_at_k_per_problem(correctness_records, k)
    return float(np.mean(per_problem)) if per_problem else 0.0


def bootstrap_pass_at_k_ci(
    correctness_records: Sequence[Sequence[int]],
    k: int,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for mean pass@k over problems.

    Resamples PROBLEMS with replacement (the unit of inference); per
    problem we reuse the empirical correctness counts. This is the
    paired-bootstrap that powers the headline ``Δ pass@1024`` claim
    (proposal §5.6).

    Returns (mean, lower, upper).
    """
    rng = random.Random(seed)
    per_problem = pass_at_k_per_problem(correctness_records, k)
    if not per_problem:
        return (0.0, 0.0, 0.0)

    means = []
    indices = list(range(len(per_problem)))
    for _ in range(n_bootstrap):
        sample = [per_problem[rng.choice(indices)] for _ in indices]
        means.append(np.mean(sample))
    means_arr = np.array(means)
    alpha = 1.0 - confidence
    lo, hi = np.quantile(means_arr, [alpha / 2, 1 - alpha / 2])
    return (float(np.mean(per_problem)), float(lo), float(hi))


def paired_bootstrap_delta(
    a_records: Sequence[Sequence[int]],
    b_records: Sequence[Sequence[int]],
    k: int,
    *,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float, float]:
    """Paired-bootstrap confidence interval for `mean(a@k) - mean(b@k)`.

    The two record sequences must be paired by problem index. Returns
    (delta_mean, lo, hi, p_value_two_sided) where ``p_value_two_sided``
    is the bootstrap proportion of resamples with delta on the
    *opposite* side of zero from the point estimate (a permutation-style
    proxy; not the correct test for tiny n but adequate for the 700-
    problem pool described in proposal §5.6).
    """
    if len(a_records) != len(b_records):
        raise ValueError(
            "paired bootstrap requires equal-length records; got "
            f"{len(a_records)} vs {len(b_records)}"
        )
    rng = random.Random(seed)
    a_per = pass_at_k_per_problem(a_records, k)
    b_per = pass_at_k_per_problem(b_records, k)
    if not a_per:
        return (0.0, 0.0, 0.0, 1.0)

    deltas = []
    indices = list(range(len(a_per)))
    for _ in range(n_bootstrap):
        sample = [rng.choice(indices) for _ in indices]
        d = np.mean([a_per[i] - b_per[i] for i in sample])
        deltas.append(d)
    deltas_arr = np.array(deltas)
    point = float(np.mean(np.array(a_per) - np.array(b_per)))
    alpha = 1.0 - confidence
    lo, hi = np.quantile(deltas_arr, [alpha / 2, 1 - alpha / 2])
    if point >= 0:
        p_two_sided = 2.0 * float(np.mean(deltas_arr <= 0))
    else:
        p_two_sided = 2.0 * float(np.mean(deltas_arr >= 0))
    p_two_sided = min(1.0, p_two_sided)
    return (point, float(lo), float(hi), p_two_sided)


def mean_pass_at_k_naive_plugin(records: Sequence[Sequence[int]], k: int) -> float:
    """The plug-in estimator ``1 - (1 - p̂)^k`` — biased; for diagnostics
    only. Provided so downstream code can compare the two estimators
    side-by-side in the camera-ready appendix."""
    if not records:
        return 0.0
    vals = []
    for r in records:
        n = len(r)
        if n == 0:
            continue
        p_hat = sum(r) / n
        vals.append(1.0 - (1.0 - p_hat) ** k)
    return float(np.mean(vals)) if vals else 0.0


def yue_crossover_check(
    base_records: Sequence[Sequence[int]],
    method_records: Sequence[Sequence[int]],
    *,
    ks: Iterable[int] = (1, 8, 64, 1024),
) -> dict[int, dict[str, float]]:
    """Per-k ``method >= base`` check (Yue et al. crossover figure).

    Returns ``{k: {"base": …, "method": …, "delta": …}}``. The
    headline-figure claim is ``method["pass@k"] >= base["pass@k"]`` for
    every k.
    """
    out: dict[int, dict[str, float]] = {}
    for k in ks:
        base = mean_pass_at_k(base_records, k)
        meth = mean_pass_at_k(method_records, k)
        out[int(k)] = {"base": base, "method": meth, "delta": meth - base}
    return out

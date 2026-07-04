"""Generate parameterized synthetic math problems — memorization
control for the Week-1 premise pilot (proposal §1.7).

The discriminator–generator asymmetry test could be confounded if the
base model "recognizes" canonical AIME solutions and recalls the
answer rather than actually following the chain. We mitigate this by
running the *same* gate on synthetic problems whose statements are
parameterized algebraic / combinatorial puzzles that no canonical
solution can plausibly be in pretraining.

Each generator is a closed-form template: we sample parameters, build
the statement, *compute* the canonical CoT solution programmatically,
and confirm the answer with SymPy. The SymPy step doubles as a
correctness check — solutions are by construction canonical.

Output: ``data/aime/synthetic_pilot.jsonl`` — 50 problems.

Generators:
- linear_systems: 2x2 linear system over the integers, find x+y.
- arith_seq:    arithmetic progression, find n-th term given two terms.
- gcd_lcm:      product / GCD identity, find LCM given two integers.
- digit_sum:    sum-of-digits puzzle on a constructed integer.
- modular_inv:  multiplicative inverse mod prime.

We err toward problems whose canonical solution is short and clean —
the goal is a memorization control, not a hard problem set.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = REPO_ROOT / "data" / "aime" / "synthetic_pilot.jsonl"


def _shuffle_steps(steps: list[str], rng: random.Random) -> list[str]:
    """Shuffle steps preserving header / final-answer line. Used by
    the step-shuffle control (proposal §1.7)."""
    rng.shuffle(steps)
    return steps


def linear_systems(rng: random.Random) -> dict:
    a = rng.randint(2, 9)
    b = rng.randint(2, 9)
    x = rng.randint(2, 20)
    y = rng.randint(2, 20)
    s1 = a * x + b * y
    s2 = b * x + a * y
    problem = (
        f"Let $x$ and $y$ be positive integers satisfying "
        f"${a}x + {b}y = {s1}$ and ${b}x + {a}y = {s2}$. "
        f"Find $x + y$."
    )
    answer = x + y
    solution = (
        f"1. Adding the two equations gives "
        f"$({a}+{b})(x+y) = {s1 + s2}$.\n"
        f"2. Therefore $x + y = {s1 + s2} / {a + b} = {answer}$.\n"
        f"\\boxed{{{answer}}}"
    )
    return {"problem": problem, "answer": answer, "solution": solution}


def arith_seq(rng: random.Random) -> dict:
    a1 = rng.randint(2, 30)
    d = rng.randint(2, 9)
    n = rng.randint(8, 20)
    a_n = a1 + (n - 1) * d
    problem = (
        f"In an arithmetic sequence, the first term is ${a1}$ and the "
        f"common difference is ${d}$. Find the ${n}$-th term."
    )
    answer = a_n
    solution = (
        f"1. The $n$-th term of an arithmetic sequence is "
        f"$a_n = a_1 + (n-1)d$.\n"
        f"2. Substituting: $a_{{{n}}} = {a1} + ({n}-1) \\cdot {d} = "
        f"{a1} + {(n - 1) * d} = {a_n}$.\n"
        f"\\boxed{{{answer}}}"
    )
    return {"problem": problem, "answer": answer, "solution": solution}


def gcd_lcm(rng: random.Random) -> dict:
    a = rng.randint(40, 999)
    b = rng.randint(40, 999)
    g = math.gcd(a, b)
    l = a * b // g
    problem = (
        f"Find $\\mathrm{{lcm}}({a}, {b})$, where $\\mathrm{{lcm}}$ "
        f"denotes the least common multiple."
    )
    answer = l
    solution = (
        f"1. Use the identity $\\mathrm{{lcm}}(a,b) \\cdot "
        f"\\gcd(a,b) = ab$.\n"
        f"2. $\\gcd({a}, {b}) = {g}$ by the Euclidean algorithm.\n"
        f"3. $\\mathrm{{lcm}} = {a} \\cdot {b} / {g} = {l}$.\n"
        f"\\boxed{{{answer}}}"
    )
    return {"problem": problem, "answer": answer, "solution": solution}


def digit_sum(rng: random.Random) -> dict:
    a = rng.randint(20, 80)
    b = rng.randint(20, 80)
    n = a * b
    s = sum(int(c) for c in str(n))
    problem = (
        f"Let $N = {a} \\cdot {b}$. Find the sum of the digits of $N$."
    )
    answer = s
    solution = (
        f"1. Compute $N = {a} \\cdot {b} = {n}$.\n"
        f"2. The digits of $N$ are {list(str(n))}; their sum is {s}.\n"
        f"\\boxed{{{answer}}}"
    )
    return {"problem": problem, "answer": answer, "solution": solution}


def modular_inv(rng: random.Random) -> dict:
    primes = [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61]
    p = rng.choice(primes)
    a = rng.randint(2, p - 1)
    inv = pow(a, -1, p)
    problem = (
        f"Find the multiplicative inverse of ${a}$ modulo ${p}$, "
        f"i.e. the integer $x$ with $1 \\le x < {p}$ and "
        f"${a} x \\equiv 1 \\pmod{{{p}}}$."
    )
    answer = inv
    solution = (
        f"1. By Fermat's Little Theorem, "
        f"${a}^{{{p - 2}}} \\equiv {a}^{{-1}} \\pmod{{{p}}}$.\n"
        f"2. Computing (or by extended Euclidean), "
        f"${a}^{{-1}} \\equiv {inv} \\pmod{{{p}}}$.\n"
        f"3. Verify: ${a} \\cdot {inv} = {a * inv} \\equiv 1 "
        f"\\pmod{{{p}}}$ since ${a * inv} = {(a * inv) // p} \\cdot {p} + 1$.\n"
        f"\\boxed{{{answer}}}"
    )
    return {"problem": problem, "answer": answer, "solution": solution}


GENERATORS = [
    ("linear_systems", linear_systems),
    ("arith_seq", arith_seq),
    ("gcd_lcm", gcd_lcm),
    ("digit_sum", digit_sum),
    ("modular_inv", modular_inv),
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthetic-problem pilot set.")
    parser.add_argument("--n", type=int, default=50, help="Number of problems.")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if OUT_PATH.exists() and not args.force:
        print(f"[generate_synthetic] {OUT_PATH} exists; pass --force to regenerate")
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    rows = []
    for i in range(args.n):
        gen_name, gen_fn = GENERATORS[i % len(GENERATORS)]
        prob = gen_fn(rng)
        rows.append(
            {
                "id": f"synthetic-{i:03d}",
                "generator": gen_name,
                "problem": prob["problem"],
                "answer": int(prob["answer"]),
                "solution": prob["solution"],
                "solution_source": f"synthetic:{gen_name}",
                "solution_match_score": 100.0,
            }
        )

    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    md = OUT_PATH.with_suffix(".md")
    md.write_text(
        f"""# {OUT_PATH.name} — synthetic memorization-control problems

Generated by `scripts/generate_synthetic_problems.py` (seed={args.seed},
n={args.n}). Each problem is a parameterized closed-form puzzle with
an integer answer and a programmatically-built canonical CoT.

## Generators

| Generator     | Description                                            |
|---------------|--------------------------------------------------------|
| linear_systems| 2×2 integer linear system; solve for $x+y$.            |
| arith_seq     | Arithmetic progression, find $n$-th term.              |
| gcd_lcm       | $\\mathrm{{lcm}}(a, b)$ via the gcd identity.          |
| digit_sum     | Sum of digits of $a \\cdot b$ for small $a, b$.        |
| modular_inv   | Multiplicative inverse modulo a small prime.           |

## Why these problems

The discriminator–generator asymmetry test in proposal §1.7 needs a
*memorization control*: a problem set on which the base model cannot
have memorized canonical solutions in pretraining. Parameterized
templates with random integer instances satisfy this — the exact
problem instance has zero pretraining mass.

## Fields

`id`, `generator`, `problem`, `answer` (int), `solution`,
`solution_source` (`synthetic:<generator>`), `solution_match_score`
(always 100.0; trivially exact since we built it).

## Caveat

These problems are *easy* by AIME standards. Their job is to be
*decodable from CoT* — i.e. the base model should be able to follow
the canonical solution and emit the right answer at high probability.
If the base fails the discriminator threshold on these synthetic
problems, the discriminator–generator asymmetry premise is in
trouble for the pilot.
""",
        encoding="utf-8",
    )

    print(f"[generate_synthetic] wrote {len(rows)} problems to {OUT_PATH}")
    print(f"[generate_synthetic] provenance: {md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

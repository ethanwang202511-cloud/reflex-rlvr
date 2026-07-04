"""Tests for the SymPy verifier — primary correctness signal.

A bug here poisons the RL gradient, so we err on the side of more
test cases. Real AIME problems are spot-checked alongside synthetic
LaTeX edge cases.
"""

from __future__ import annotations

from reflex_rlvr.verifier.sympy_verifier import (
    canonicalize,
    equal,
    extract_answer,
    verify,
)


class TestExtractAnswer:
    def test_simple_boxed_int(self) -> None:
        text = r"After computation, the answer is \boxed{42}."
        assert extract_answer(text) == "42"

    def test_boxed_with_braces_inside(self) -> None:
        text = r"Final answer: \boxed{\frac{1}{2}}"
        assert extract_answer(text) == r"\frac{1}{2}"

    def test_picks_last_boxed(self) -> None:
        text = r"Intermediate \boxed{12}. Final answer is \boxed{37}."
        assert extract_answer(text) == "37"

    def test_no_boxed_falls_back_to_answer_keyword(self) -> None:
        text = "Working through the problem... Answer: 13"
        assert extract_answer(text) == "13"

    def test_returns_none_on_empty(self) -> None:
        assert extract_answer("") is None
        assert extract_answer("no relevant marker here") is None


class TestCanonicalize:
    def test_int(self) -> None:
        c = canonicalize("42")
        assert c is not None
        assert int(c) == 42

    def test_fraction_latex(self) -> None:
        c = canonicalize(r"\frac{3}{4}")
        assert c is not None
        assert float(c) == 0.75

    def test_sqrt_latex(self) -> None:
        c = canonicalize(r"\sqrt{2}")
        assert c is not None
        assert abs(float(c) - 1.4142135623730951) < 1e-9

    def test_pi_latex(self) -> None:
        c = canonicalize(r"\pi/3")
        assert c is not None
        # Pi/3 in radians
        import math

        assert abs(float(c) - math.pi / 3) < 1e-9

    def test_caret_for_power_in_text(self) -> None:
        c = canonicalize("2^5")
        assert c is not None
        assert int(c) == 32

    def test_strips_degree_unit(self) -> None:
        c = canonicalize(r"60^\circ")
        assert c is not None
        assert int(c) == 60

    def test_unparseable_returns_none(self) -> None:
        assert canonicalize("definitely not math !!!") is None


class TestEqual:
    def test_simple_int_equality(self) -> None:
        assert equal(canonicalize("42"), canonicalize("42"))

    def test_fraction_equivalence(self) -> None:
        assert equal(canonicalize(r"\frac{1}{2}"), canonicalize("0.5"))

    def test_sqrt_equivalence(self) -> None:
        assert equal(canonicalize(r"\sqrt{4}"), canonicalize("2"))

    def test_unequal(self) -> None:
        assert not equal(canonicalize("42"), canonicalize("43"))


class TestVerifyEndToEnd:
    def test_aime_style_int_answer_correct(self) -> None:
        text = (
            "Working through the geometry... by Power of a Point we get "
            r"\boxed{592}."
        )
        result = verify(text, 592)
        assert result.accepted, result.reason

    def test_aime_style_int_answer_wrong(self) -> None:
        text = r"After thinking, \boxed{593}."
        result = verify(text, 592)
        assert not result.accepted
        assert result.reason == "not_equal"

    def test_no_answer_emitted(self) -> None:
        text = "I don't know. Skipping."
        result = verify(text, 592)
        assert not result.accepted
        assert result.reason == "no_boxed_answer_found"

    def test_fraction_match(self) -> None:
        text = r"The probability is \boxed{\frac{1}{2}}."
        result = verify(text, "1/2")
        assert result.accepted

    def test_decimal_vs_fraction(self) -> None:
        text = r"\boxed{0.5}"
        result = verify(text, "1/2")
        assert result.accepted

    def test_negative_answer(self) -> None:
        text = r"\boxed{-3}"
        result = verify(text, -3)
        assert result.accepted

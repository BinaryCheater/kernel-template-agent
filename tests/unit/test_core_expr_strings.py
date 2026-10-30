from __future__ import annotations

import pytest

from kta.core.expr import Expr, ExprError


def test_string_equality_and_membership() -> None:
    e = Expr.parse("mechanism == 'cp.async' and level != 'reg'", {"mechanism", "level"})
    assert not e.is_linear
    assert e.eval({"mechanism": "cp.async", "level": "smem"}) is True
    m = Expr.parse("level in ['smem', 'reg'] and BM <= 128", {"level", "BM"})
    assert not m.is_linear
    assert m.eval({"level": "reg", "BM": 64}) is True
    assert Expr.parse("level not in ('smem',)", {"level"}).eval({"level": "smem"}) is False


@pytest.mark.parametrize(
    "src",
    [
        "level < 'smem'",  # ordering on strings
        "level + 'x' == 'y'",  # string arithmetic
        "level in levels",  # membership needs a literal list
        "level in []",
        "level in [other]",
        "a in [1] in [True]",
        "'x'",
    ],
)
def test_string_misuse_rejected(src: str) -> None:
    with pytest.raises(ExprError):
        Expr.parse(src, {"level", "levels", "other", "a"})


def test_numeric_expressions_unchanged() -> None:
    e = Expr.parse("2 * BM + BK <= hw.smem_per_cta", {"BM", "BK"})
    assert e.is_linear
    assert e.eval({"BM": 64, "BK": 32}, hw={"smem_per_cta": 1000}) is True

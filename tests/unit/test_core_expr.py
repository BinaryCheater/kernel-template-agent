from __future__ import annotations

from typing import Any

import pytest

from kta.core.expr import Expr, ExprError

DECLARED = {"x", "y", "stages"}


@pytest.mark.parametrize(
    ("src", "env", "expected"),
    [
        ("x + y * 2", {"x": 1, "y": 3}, 7),
        ("x // 2 + x % 3", {"x": 7}, 4),
        ("x / 2", {"x": 3}, 1.5),
        ("2 ** stages", {"stages": 3}, 8),
        ("min(x, y) + max(x, y) + abs(-x)", {"x": 1, "y": 5}, 7),
        ("ceil(x / 2) + floor(x / 2)", {"x": 5}, 5),
        ("ceil_div(x, 4)", {"x": 9}, 3),
        ("log2(x)", {"x": 8}, 3.0),
        ("is_pow2(x) and not is_pow2(y)", {"x": 8, "y": 6}, True),
        ("1 < x <= 3", {"x": 3}, True),
        ("1 < x <= 3", {"x": 4}, False),
        ("x if y > 0 else -x", {"x": 2, "y": -1}, -2),
        ("x or y", {"x": 0, "y": 5}, 5),
        ("stages * p.tile <= hw.smem", {"stages": 4}, True),
        ("case.m >= 1", {}, True),
    ],
)
def test_eval(src: str, env: dict[str, Any], expected: Any) -> None:
    expr = Expr.parse(src, DECLARED)
    assert expr.eval(env, hw={"smem": 1024}, p={"tile": 256}, case={"m": 4}) == expected


@pytest.mark.parametrize(
    "src",
    [
        "__import__('os')",
        "open('x')",
        "x[0]",
        "x.real",
        "lambda: 1",
        "[i for i in x]",
        "'text'",
        "z + 1",
        "hw",
        "min",
        "x in y",
        "x is y",
        "x @ y",
        "max(*x)",
        "min(x, key=y)",
        "x = 1",
        "x +",
    ],
)
def test_rejected_syntax(src: str) -> None:
    with pytest.raises(ExprError):
        Expr.parse(src, DECLARED)


def test_free_vars_and_attribute_refs() -> None:
    expr = Expr.parse("stages * p.tile_bytes <= hw.smem_per_cta + case.m", DECLARED)
    assert expr.free_vars == {"stages"}
    assert expr.attr_refs == {"p.tile_bytes", "hw.smem_per_cta", "case.m"}


@pytest.mark.parametrize(
    ("src", "linear"),
    [
        ("2 * x + 3 * y <= 10", True),
        ("x - y >= hw.a * p.b", True),
        ("hw.a * x + ceil_div(p.n, 4) == y", True),
        ("-x + 1 < 0 and y >= 1", True),
        ("x * y <= 3", False),
        ("x / 2 <= 3", False),
        ("x != 3", False),
        ("x <= 1 or y <= 1", False),
        ("min(x, y) <= 3", False),
        ("x ** 2 <= 4", False),
        ("x if y else 1", False),
    ],
)
def test_linear_subset(src: str, linear: bool) -> None:
    assert Expr.parse(src, DECLARED).is_linear is linear


def test_eval_errors_are_expr_errors() -> None:
    with pytest.raises(ExprError, match="no value"):
        Expr.parse("x + 1", DECLARED).eval({})
    with pytest.raises(ExprError, match="no value for hw"):
        Expr.parse("hw.smem", DECLARED).eval({})
    with pytest.raises(ExprError):
        Expr.parse("x / y", DECLARED).eval({"x": 1, "y": 0})
    with pytest.raises(ExprError, match="exponent"):
        Expr.parse("2 ** x", DECLARED).eval({"x": 1000})
    with pytest.raises(ExprError, match="log2"):
        Expr.parse("log2(x)", DECLARED).eval({"x": 0})


def test_expr_error_is_a_value_error() -> None:
    with pytest.raises(ValueError):
        Expr.parse("undeclared", DECLARED)

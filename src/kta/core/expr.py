"""Expression language for constraints, preconditions, scope and cost terms (spec/schemas.md §3).

A subset of Python expressions, parsed with ``ast`` and interpreted here; Python's
``eval`` is never used. Free names must be declared variables; ``hw.<name>``,
``p.<name>`` and ``case.<name>`` read the hardware contract, pattern/workload
parameters and the case.

String literals may appear only as operands of ``==``/``!=`` and as elements of the
literal list on the right of ``in``/``not in`` (for named mechanisms, levels and
instructions); any comparison involving them is non-linear.

Linear subset (exportable to MIP): additions, subtractions, multiplication where one
side is constant, comparisons other than ``!=``, and ``and``. A subexpression is
constant when it uses no declared variable (literals and hw/p/case attributes are
fixed at solve time).
"""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from kta.core.errors import SchemaError

ROOTS = ("hw", "p", "case")
MAX_POW_EXPONENT = 64
MAX_SOURCE_CHARS = 4000


class ExprError(SchemaError):
    """Syntax, free-variable or evaluation error in an expression."""


def _ceil_div(a: Any, b: Any) -> int:
    if b == 0:
        raise ExprError("ceil_div by zero")
    return -(-int(a) // int(b))


def _log2(x: Any) -> float:
    if x <= 0:
        raise ExprError(f"log2 of non-positive value {x!r}")
    return math.log2(x)


def _is_pow2(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x > 0 and x & (x - 1) == 0


FUNCTIONS: dict[str, Callable[..., Any]] = {
    "min": min,
    "max": max,
    "abs": abs,
    "ceil": math.ceil,
    "floor": math.floor,
    "ceil_div": _ceil_div,
    "log2": _log2,
    "is_pow2": _is_pow2,
}

_BINOPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: operator.mod,
    ast.Pow: lambda a, b: a**b,
}
_CMPOPS: dict[type[ast.cmpop], Callable[[Any, Any], bool]] = {
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
}
_LINEAR_CMP = (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq)
_EQUALITY = (ast.Eq, ast.NotEq)
_MEMBERSHIP = (ast.In, ast.NotIn)


def _attr_ref(node: ast.Attribute) -> str:
    if not isinstance(node.value, ast.Name) or node.value.id not in ROOTS:
        raise ExprError("attribute access is only allowed as hw.<name>, p.<name>, case.<name>")
    return f"{node.value.id}.{node.attr}"


class _Checker:
    """Validates the allowed node set and collects names."""

    def __init__(self, declared: set[str]) -> None:
        self.declared = declared
        self.vars: set[str] = set()
        self.attrs: set[str] = set()

    def visit(self, node: ast.expr, allow_str: bool = False) -> None:
        match node:
            case ast.Constant(value=value):
                if isinstance(value, str) and allow_str:
                    return
                if not isinstance(value, (bool, int, float)):
                    raise ExprError(
                        f"literal {value!r} is not a number or boolean "
                        "(strings only in ==, != and `in [...]`)"
                    )
            case ast.Name(id=name):
                if name in FUNCTIONS or name in ROOTS:
                    raise ExprError(f"{name!r} cannot be used as a value")
                if name not in self.declared:
                    raise ExprError(f"undeclared variable {name!r}")
                self.vars.add(name)
            case ast.Attribute():
                self.attrs.add(_attr_ref(node))
            case ast.BinOp(left=left, op=op, right=right):
                if type(op) not in _BINOPS:
                    raise ExprError(f"operator {type(op).__name__} is not allowed")
                self.visit(left)
                self.visit(right)
            case ast.UnaryOp(op=op, operand=operand):
                if not isinstance(op, (ast.USub, ast.UAdd, ast.Not)):
                    raise ExprError(f"operator {type(op).__name__} is not allowed")
                self.visit(operand)
            case ast.BoolOp(values=values):
                for v in values:
                    self.visit(v)
            case ast.Compare(left=left, ops=ops, comparators=comparators):
                self._compare(left, ops, comparators)
            case ast.IfExp(test=test, body=body, orelse=orelse):
                self.visit(test)
                self.visit(body)
                self.visit(orelse)
            case ast.Call(func=func, args=args, keywords=keywords):
                if not isinstance(func, ast.Name) or func.id not in FUNCTIONS:
                    raise ExprError(f"only the functions {sorted(FUNCTIONS)} may be called")
                if keywords:
                    raise ExprError("keyword arguments are not allowed")
                for a in args:
                    if isinstance(a, ast.Starred):
                        raise ExprError("argument unpacking is not allowed")
                    self.visit(a)
            case _:
                raise ExprError(f"{type(node).__name__} is not allowed in expressions")

    def _compare(self, left: ast.expr, ops: list[ast.cmpop], comparators: list[ast.expr]) -> None:
        if any(isinstance(o, _MEMBERSHIP) for o in ops):
            if len(ops) != 1:
                raise ExprError("`in` / `not in` cannot be chained")
            target = comparators[0]
            if not isinstance(target, ast.List | ast.Tuple) or not target.elts:
                raise ExprError("`in` needs a non-empty literal list, e.g. x in ['a', 'b']")
            self.visit(left, allow_str=True)
            for elt in target.elts:
                if not isinstance(elt, ast.Constant):
                    raise ExprError("`in` lists may only contain literals")
                self.visit(elt, allow_str=True)
            return
        if any(type(o) not in _CMPOPS for o in ops):
            raise ExprError("only < <= > >= == != in comparisons are allowed")
        equality_only = all(isinstance(o, _EQUALITY) for o in ops)
        for operand in [left, *comparators]:
            self.visit(operand, allow_str=equality_only)


def _is_const(node: ast.expr) -> bool:
    """True when no declared variable occurs (function names and hw/p/case roots are not)."""
    match node:
        case ast.Name():
            return False
        case ast.Attribute():
            return True
        case ast.Call(args=args):
            return all(_is_const(a) for a in args)
        case _:
            return all(_is_const(c) for c in ast.iter_child_nodes(node) if isinstance(c, ast.expr))


def _has_str(node: ast.expr) -> bool:
    return any(isinstance(n, ast.Constant) and isinstance(n.value, str) for n in ast.walk(node))


def _linear(node: ast.expr) -> bool:
    if _is_const(node):
        return True
    match node:
        case ast.Name():
            return True
        case ast.BinOp(left=left, op=ast.Add() | ast.Sub(), right=right):
            return _linear(left) and _linear(right)
        case ast.BinOp(left=left, op=ast.Mult(), right=right):
            return (_is_const(left) and _linear(right)) or (_is_const(right) and _linear(left))
        case ast.UnaryOp(op=ast.USub() | ast.UAdd(), operand=operand):
            return _linear(operand)
        case ast.Compare(left=left, ops=ops, comparators=comparators):
            if any(_has_str(x) for x in [left, *comparators]):
                return False
            return all(isinstance(o, _LINEAR_CMP) for o in ops) and all(
                _linear(x) for x in [left, *comparators]
            )
        case ast.BoolOp(op=ast.And(), values=values):
            return all(_linear(v) for v in values)
        case _:
            return False


@dataclass(frozen=True)
class Expr:
    src: str
    free_vars: frozenset[str]
    attr_refs: frozenset[str]  # e.g. {"hw.smem_per_cta", "p.n"}
    is_linear: bool
    _tree: ast.expr

    @staticmethod
    def parse(src: str, declared: set[str]) -> Expr:
        if len(src) > MAX_SOURCE_CHARS:
            raise ExprError(f"expression longer than {MAX_SOURCE_CHARS} characters")
        try:
            tree = ast.parse(src.strip(), mode="eval").body
        except SyntaxError as exc:
            raise ExprError(f"syntax error in {src!r}: {exc.msg}") from exc
        checker = _Checker(declared)
        checker.visit(tree)
        return Expr(
            src=src,
            free_vars=frozenset(checker.vars),
            attr_refs=frozenset(checker.attrs),
            is_linear=_linear(tree),
            _tree=tree,
        )

    def eval(
        self,
        env: Mapping[str, Any],
        hw: Mapping[str, Any] | None = None,
        p: Mapping[str, Any] | None = None,
        case: Mapping[str, Any] | None = None,
    ) -> Any:
        scopes: dict[str, Mapping[str, Any]] = {"hw": hw or {}, "p": p or {}, "case": case or {}}
        try:
            return _Evaluator(env, scopes).run(self._tree)
        except ExprError:
            raise
        except (ArithmeticError, TypeError, ValueError) as exc:
            raise ExprError(f"cannot evaluate {self.src!r}: {exc}") from exc


class _Evaluator:
    def __init__(self, env: Mapping[str, Any], scopes: dict[str, Mapping[str, Any]]) -> None:
        self.env = env
        self.scopes = scopes

    def run(self, node: ast.expr) -> Any:
        match node:
            case ast.Constant(value=value):
                return value
            case ast.Name(id=name):
                if name not in self.env:
                    raise ExprError(f"no value for variable {name!r}")
                return self.env[name]
            case ast.Attribute(value=ast.Name(id=root), attr=attr):
                scope = self.scopes[root]
                if attr not in scope:
                    raise ExprError(f"no value for {root}.{attr}")
                return scope[attr]
            case ast.BinOp(left=left, op=op, right=right):
                return self._binop(op, self.run(left), self.run(right))
            case ast.UnaryOp(op=op, operand=operand):
                value = self.run(operand)
                if isinstance(op, ast.Not):
                    return not value
                return -value if isinstance(op, ast.USub) else +value
            case ast.BoolOp(op=op, values=values):
                return self._boolop(isinstance(op, ast.And), values)
            case ast.Compare(left=left, ops=ops, comparators=comparators):
                return self._compare(left, ops, comparators)
            case ast.IfExp(test=test, body=body, orelse=orelse):
                return self.run(body) if self.run(test) else self.run(orelse)
            case ast.Call(func=ast.Name(id=name), args=args):
                return FUNCTIONS[name](*[self.run(a) for a in args])
            case _:
                raise ExprError(f"{type(node).__name__} cannot be evaluated")

    def _binop(self, op: ast.operator, left: Any, right: Any) -> Any:
        if isinstance(op, ast.Pow) and abs(right) > MAX_POW_EXPONENT:
            raise ExprError(f"exponent {right!r} exceeds {MAX_POW_EXPONENT}")
        return _BINOPS[type(op)](left, right)

    def _boolop(self, is_and: bool, values: list[ast.expr]) -> Any:
        result: Any = None
        for v in values:
            result = self.run(v)
            if bool(result) != is_and:
                return result
        return result

    def _compare(self, left: ast.expr, ops: list[ast.cmpop], comparators: list[ast.expr]) -> bool:
        current = self.run(left)
        if isinstance(ops[0], _MEMBERSHIP):
            target = comparators[0]
            if not isinstance(target, ast.List | ast.Tuple):
                raise ExprError("`in` needs a literal list")
            found = current in [self.run(e) for e in target.elts]
            return found if isinstance(ops[0], ast.In) else not found
        for op, nxt in zip(ops, comparators, strict=True):
            value = self.run(nxt)
            if not _CMPOPS[type(op)](current, value):
                return False
            current = value
        return True

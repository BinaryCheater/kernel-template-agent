"""Condition builder: expressions of the KTA language (spec/schemas.md §3) as Z3 terms.

Design variables get their declared domains; ``p.*``, ``hw.*`` and ``case.*`` are free
symbols unless the caller binds them. Sorts are inferred from use: an attribute used as a
truth value is Bool, one compared with string literals is an enumeration over those
literals plus one value standing for "anything else", the rest are Int. Python's floor
division and modulo are encoded exactly (also for negative divisors); ``/`` is real
division. A construct without a faithful encoding (``log2`` of a variable, a variable
exponent) raises ``Unsupported``, which the caller reports as ``unresolved``.

This module is trusted: it is one third of the trusted base (with the replay interpreter
and Z3), so it stays small and literal.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import z3

from kta.core.schemas.knowledge import RangeSpec, VarSpec

OTHER = "•other"  # the enumeration value no literal names
MAX_POW2_EXP = 62
MAX_ENUMERATED_RANGE = 4096


class Unsupported(Exception):
    """No faithful encoding; the obligation stays unresolved with this reason."""


@dataclass(frozen=True)
class Sym:
    name: str
    term: Any  # z3 term
    kind: str  # int | bool | enum | real
    enum: tuple[str, ...] = ()

    def decode(self, v: Any) -> Any:
        if self.kind == "enum":
            i = int(v)
            return self.enum[i] if 0 <= i < len(self.enum) else OTHER
        return v


@dataclass(frozen=True)
class _V:
    term: Any
    kind: str  # int | real | bool | enum | lit
    sym: Sym | None = None
    lit: str | None = None


def _name_of(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return f"{node.value.id}.{node.attr}"
    return None


def parse(src: str) -> ast.expr:
    try:
        return ast.parse(src.strip(), mode="eval").body
    except SyntaxError as exc:
        raise Unsupported(f"cannot parse {src!r}: {exc.msg}") from exc


# ------------------------------------------------------------------ sort inference


def infer(
    srcs: Iterable[str], values: Iterable[str] = ()
) -> dict[str, tuple[str, tuple[str, ...]]]:
    """name -> (kind, string literals) for every name used in ``srcs`` (truth-valued
    expressions) and ``values`` (value expressions such as chain bindings)."""
    bools: set[str] = set()
    lits: dict[str, set[str]] = {}
    seen: set[str] = set()

    def as_bool(node: ast.expr) -> None:
        n = _name_of(node)
        if n:
            bools.add(n)

    def visit(node: ast.expr, truth: bool) -> None:
        n = _name_of(node)
        if n:
            seen.add(n)
            if truth:
                bools.add(n)
            return
        if isinstance(node, ast.BoolOp):
            for v in node.values:
                visit(v, True)
        elif isinstance(node, ast.UnaryOp):
            visit(node.operand, isinstance(node.op, ast.Not))
        elif isinstance(node, ast.IfExp):
            visit(node.test, True)
            visit(node.body, truth)
            visit(node.orelse, truth)
        elif isinstance(node, ast.Compare):
            items = [node.left, *node.comparators]
            strs = [c.value for c in ast.walk(node) if isinstance(c, ast.Constant)]
            strs = [s for s in strs if isinstance(s, str)]
            bool_consts = any(
                isinstance(c, ast.Constant) and isinstance(c.value, bool) for c in items
            )
            for it in items:
                nm = _name_of(it)
                if nm and strs:
                    lits.setdefault(nm, set()).update(strs)
                if nm and bool_consts:
                    as_bool(it)
                visit(it, False)
        elif isinstance(node, ast.BinOp):
            visit(node.left, False)
            visit(node.right, False)
        elif isinstance(node, ast.Call):
            for a in node.args:
                visit(a, False)
        elif isinstance(node, ast.List | ast.Tuple):
            for e in node.elts:
                visit(e, False)

    for s in srcs:
        visit(parse(s), True)
    for s in values:
        visit(parse(s), False)
    out: dict[str, tuple[str, tuple[str, ...]]] = {}
    for n in seen:
        if n in lits:
            out[n] = ("enum", tuple(sorted(lits[n])))
        elif n in bools:
            out[n] = ("bool", ())
        else:
            out[n] = ("int", ())
    return out


# ------------------------------------------------------------------ symbols and domains


def _values(d: list[Any] | RangeSpec) -> list[Any] | None:
    if isinstance(d, list):
        return list(d)
    if d.pow2 or (d.hi - d.lo) // max(d.step, 1) <= MAX_ENUMERATED_RANGE:
        vals = list(range(d.lo, d.hi + 1, max(d.step, 1)))
        return [v for v in vals if v > 0 and v & (v - 1) == 0] if d.pow2 else vals
    return None


def domain_values(v: VarSpec) -> list[Any] | None:
    """The finite list of values, or None for a wide range (kept symbolic)."""
    return _values(v.domain)


def declare(v: VarSpec, extra_lits: Iterable[str] = ()) -> tuple[Sym, Any]:
    """The symbol of a design variable and its domain constraint."""
    d = v.domain
    if isinstance(d, list):
        if d and all(isinstance(x, bool) for x in d):
            t = z3.Bool(v.name)
            allowed = {bool(x) for x in d}
            return Sym(v.name, t, "bool"), z3.BoolVal(True) if len(allowed) == 2 else (
                t if True in allowed else z3.Not(t)
            )
        if d and all(isinstance(x, str) for x in d):
            names = tuple(str(x) for x in d) + tuple(
                x for x in sorted(set(extra_lits)) if x not in d
            )
            t = z3.Int(v.name)
            return Sym(v.name, t, "enum", names), z3.And(t >= 0, t < len(d))
        if d and all(isinstance(x, int) and not isinstance(x, bool) for x in d):
            t = z3.Int(v.name)
            return Sym(v.name, t, "int"), z3.Or(*[t == int(x) for x in d])
        raise Unsupported(f"variable {v.name}: mixed or empty domain {d!r}")
    t: Any = z3.Int(v.name)
    c: list[Any] = [t >= d.lo, t <= d.hi]
    if d.step > 1:
        c.append((t - d.lo) % d.step == 0)
    if d.pow2:
        c.append(z3.Or(*[t == 2**i for i in range(MAX_POW2_EXP) if d.lo <= 2**i <= d.hi]))
    return Sym(v.name, t, "int"), z3.And(*c)


def free(name: str, kind: str, lits: tuple[str, ...] = ()) -> Sym:
    if kind == "bool":
        return Sym(name, z3.Bool(name), "bool")
    if kind == "enum":
        return Sym(name, z3.Int(name), "enum", (*lits, OTHER))
    return Sym(name, z3.Int(name), "int")


def enum_range(s: Sym) -> Any:
    return z3.And(s.term >= 0, s.term < len(s.enum)) if s.kind == "enum" else z3.BoolVal(True)


def constant(name: str, value: Any, like: Sym | None = None) -> Sym:
    """A symbol fixed to a value (chain bindings, restricted domains)."""
    if isinstance(value, bool):
        return Sym(name, z3.BoolVal(value), "bool")
    if isinstance(value, int):
        return Sym(name, z3.IntVal(value), "int")
    if isinstance(value, float):
        return Sym(name, z3.RealVal(value), "real")
    if isinstance(value, str) and like is not None and like.kind == "enum":
        idx = like.enum.index(value) if value in like.enum else len(like.enum)
        return Sym(name, z3.IntVal(idx), "enum", like.enum)
    if isinstance(value, str):
        return Sym(name, z3.IntVal(0), "enum", (value,))
    raise Unsupported(f"{name}: cannot fix to {value!r}")


# ------------------------------------------------------------------ translation


def _floordiv(a: Any, b: Any) -> Any:
    # z3 integer division is Euclidean; Python's // floors. Equal for b > 0.
    return z3.If(b > 0, a / b, (-a) / (-b))


def _num(v: _V) -> Any:
    if v.kind == "bool":
        return z3.If(v.term, 1, 0)
    if v.kind in ("enum", "lit"):
        raise Unsupported("arithmetic on a string value")
    return v.term


def _real(v: _V) -> Any:
    t = _num(v)
    return z3.ToReal(t) if v.kind == "int" or v.kind == "bool" else t


def _boolean(v: _V) -> Any:
    if v.kind == "bool":
        return v.term
    if v.kind in ("int", "real"):
        return v.term != 0
    raise Unsupported("a string value used as a truth value")


class Translator:
    def __init__(self, env: Mapping[str, Sym]) -> None:
        self.env = env

    def bool(self, src: str) -> Any:
        return _boolean(self._go(parse(src)))

    def num(self, src: str) -> Any:
        return _num(self._go(parse(src)))

    def value(self, src: str) -> tuple[Any, str]:
        """(term, kind) of an expression in its own sort: int, real, bool or enum."""
        v = self._go(parse(src))
        if v.kind == "lit":
            raise Unsupported(f"bare string {src!r} is not a value here")
        return v.term, v.kind

    def _go(self, node: ast.expr) -> _V:
        n = _name_of(node)
        if n is not None:
            s = self.env.get(n)
            if s is None:
                raise Unsupported(f"unbound name {n}")
            return _V(s.term, s.kind, s)
        if isinstance(node, ast.Constant):
            c = node.value
            if isinstance(c, bool):
                return _V(z3.BoolVal(c), "bool")
            if isinstance(c, int):
                return _V(z3.IntVal(c), "int")
            if isinstance(c, float):
                return _V(z3.RealVal(c), "real")
            if isinstance(c, str):
                return _V(None, "lit", lit=c)
            raise Unsupported(f"constant {c!r}")
        if isinstance(node, ast.UnaryOp):
            v = self._go(node.operand)
            if isinstance(node.op, ast.Not):
                return _V(z3.Not(_boolean(v)), "bool")
            if isinstance(node.op, ast.USub):
                return _V(-_num(v), v.kind if v.kind != "bool" else "int")
            if isinstance(node.op, ast.UAdd):
                return v
            raise Unsupported("unary operator")
        if isinstance(node, ast.BoolOp):
            parts = [_boolean(self._go(x)) for x in node.values]
            return _V(z3.And(*parts) if isinstance(node.op, ast.And) else z3.Or(*parts), "bool")
        if isinstance(node, ast.BinOp):
            return self._binop(node)
        if isinstance(node, ast.Compare):
            return self._compare(node)
        if isinstance(node, ast.IfExp):
            t, a, b = self._go(node.test), self._go(node.body), self._go(node.orelse)
            if a.kind == "bool" and b.kind == "bool":
                return _V(z3.If(_boolean(t), a.term, b.term), "bool")
            kind = "real" if "real" in (a.kind, b.kind) else "int"
            f = _real if kind == "real" else _num
            return _V(z3.If(_boolean(t), f(a), f(b)), kind)
        if isinstance(node, ast.Call):
            return self._call(node)
        raise Unsupported(f"construct {type(node).__name__}")

    def _binop(self, node: ast.BinOp) -> _V:
        a, b = self._go(node.left), self._go(node.right)
        real = "real" in (a.kind, b.kind)
        op = node.op
        if isinstance(op, ast.Div):
            return _V(_real(a) / _real(b), "real")
        if isinstance(op, ast.Add | ast.Sub | ast.Mult):
            x, y = (_real(a), _real(b)) if real else (_num(a), _num(b))
            r = x + y if isinstance(op, ast.Add) else x - y if isinstance(op, ast.Sub) else x * y
            return _V(r, "real" if real else "int")
        if real:
            raise Unsupported("floor division or modulo on real values")
        x, y = _num(a), _num(b)
        if isinstance(op, ast.FloorDiv):
            return _V(_floordiv(x, y), "int")
        if isinstance(op, ast.Mod):
            return _V(x - y * _floordiv(x, y), "int")
        if isinstance(op, ast.Pow):
            e: Any = z3.simplify(y)
            n = int(e.as_long()) if z3.is_int_value(e) else -1
            if not 0 <= n <= 8:
                raise Unsupported("exponent must be a literal 0..8")
            r: Any = z3.IntVal(1)
            for _ in range(n):
                r = r * x
            return _V(r, "int")
        raise Unsupported("binary operator")

    def _eq(self, a: _V, b: _V) -> Any:
        if a.kind == "lit" and b.kind == "lit":
            return z3.BoolVal(a.lit == b.lit)
        if b.kind == "lit":
            a, b = b, a
        if a.kind == "lit":
            if b.kind != "enum" or b.sym is None:
                raise Unsupported("string compared with a non-string value")
            names = b.sym.enum
            return b.term == names.index(a.lit) if a.lit in names else z3.BoolVal(False)
        if a.kind == "enum" or b.kind == "enum":
            return a.term == b.term
        if a.kind == "bool" or b.kind == "bool":
            return _boolean(a) == _boolean(b)
        return (_real(a) == _real(b)) if "real" in (a.kind, b.kind) else a.term == b.term

    def _compare(self, node: ast.Compare) -> _V:
        parts: list[Any] = []
        left = self._go(node.left)
        for op, comp in zip(node.ops, node.comparators, strict=True):
            if isinstance(op, ast.In | ast.NotIn):
                if not isinstance(comp, ast.List | ast.Tuple):
                    raise Unsupported("'in' needs a literal list")
                inside = z3.Or(*[self._eq(left, self._go(e)) for e in comp.elts])
                parts.append(inside if isinstance(op, ast.In) else z3.Not(inside))
                continue
            right = self._go(comp)
            if isinstance(op, ast.Eq):
                parts.append(self._eq(left, right))
            elif isinstance(op, ast.NotEq):
                parts.append(z3.Not(self._eq(left, right)))
            else:
                real = "real" in (left.kind, right.kind)
                x, y = (_real(left), _real(right)) if real else (_num(left), _num(right))
                parts.append(
                    x < y
                    if isinstance(op, ast.Lt)
                    else x <= y
                    if isinstance(op, ast.LtE)
                    else x > y
                    if isinstance(op, ast.Gt)
                    else x >= y
                )
            left = right
        return _V(z3.And(*parts) if len(parts) > 1 else parts[0], "bool")

    def _call(self, node: ast.Call) -> _V:
        if not isinstance(node.func, ast.Name):
            raise Unsupported("call of a non-name")
        f = node.func.id
        args = [self._go(a) for a in node.args]
        real = any(a.kind == "real" for a in args)
        conv = _real if real else _num
        kind = "real" if real else "int"
        if f in ("min", "max") and args:
            acc = conv(args[0])
            for a in args[1:]:
                t = conv(a)
                acc = z3.If(t < acc, t, acc) if f == "min" else z3.If(t > acc, t, acc)
            return _V(acc, kind)
        if f == "abs" and len(args) == 1:
            t = conv(args[0])
            return _V(z3.If(t < 0, -t, t), kind)
        if f == "ceil_div" and len(args) == 2:
            return _V(-_floordiv(-_num(args[0]), _num(args[1])), "int")
        if f in ("floor", "ceil") and len(args) == 1:
            if args[0].kind != "real":
                return _V(_num(args[0]), "int")
            t = args[0].term
            return _V(z3.ToInt(t) if f == "floor" else -z3.ToInt(-t), "int")
        if f == "is_pow2" and len(args) == 1:
            t = _num(args[0])
            return _V(z3.Or(*[t == 2**i for i in range(MAX_POW2_EXP)]), "bool")
        raise Unsupported(f"function {f}")

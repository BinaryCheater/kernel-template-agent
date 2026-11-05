"""Per-lane address values for plain global loads (KTA PERFMODEL round 3): the lines and sectors an
LDG touches follow from its width and its 32 lanes' addresses, not from a fixed count.

A register's value is kept as (base, offsets): `base` a sorted tuple of (symbol, coefficient) for
the warp-uniform values the replay does not know (kernel-parameter pointers by constant-bank offset,
other uniform values by the instruction that wrote them), `offsets` the 32 lanes' known parts.
Linear instructions (add, multiply by a known constant, shift left, LEA) keep a base; bitwise and
right shifts need an empty base (exact values). A register written by anything else is uniform if
every source is uniform (its lane-0 value from the warp's value tracking, else a fresh symbol) and
unknown otherwise. Loads, shuffles and the like write unknown registers. Carries between the halves
of 64-bit pairs are ignored (only the low half of an address pair is tracked); a base is assumed
128-byte aligned (cudaMalloc aligns to 256 B).

Round 15: integer instructions over exact lanes are evaluated as a class (`_alu`): every form in
compiled CUDA's census of adds, products with their high halves, shifts, LEA, PRMT, bit counts and
the float steps of a division. Bitwise functions over a base act on the bits the base leaves free
(`_lop3`). `address_gap` names what is left unresolved; it goes on the scorecard.
"""

from __future__ import annotations

import math
import re
import struct
from typing import Any

Val = tuple[tuple, tuple]  # (base, offsets[32])
M32 = 0xFFFFFFFF
LANES = tuple(range(32))
ZERO = (0,) * 32
APART = 0x40000000
WRITES_UNKNOWN = {
    "LDG",
    "LDS",
    "LD",
    "LDL",
    "LDSM",
    "SHFL",
    "ATOM",
    "ATOMG",
    "ATOMS",
    "LDGSTS",
    "MATCH",
    "VOTE",
    "REDUX",
    "F2I",
    "I2F",
    "FFMA",
    "FADD",
    "FMUL",
    "HFMA2",
    "HADD2",
    "HMUL2",
    "MUFU",
    "POPC",
    "FLO",
    "BREV",
    "I2FP",
    "F2FP",
    "FSEL",
    "FMNMX",
    "CS2R",
    "S2R",
    "S2UR",
    "HMMA",
    "QMMA",
    "IMMA",
}
NO_DEST = {
    "STG",
    "STS",
    "ST",
    "STL",
    "RED",
    "BAR",
    "BRA",
    "EXIT",
    "BSSY",
    "BSYNC",
    "WARPSYNC",
    "NOP",
    "DEPBAR",
    "MEMBAR",
    "FENCE",
    "CCTL",
    "ERRBAR",
    "RET",
    "CALL",
    "YIELD",
    "ISETP",
    "UISETP",
    "FSETP",
    "HSETP2",
    "PLOP3",
    "UPLOP3",
    "VOTEU",
    "LDGDEPBAR",
    "ARRIVES",
    "SYNCS",
    "BPT",
}
FLOAT_STEP = {"FMUL", "FADD", "FFMA", "MUFU", "F2I"}
_DEC = re.compile(r"^\d+\.\d*(e[-+]?\d+)?$|^\d+e[-+]?\d+$")
CARRY_DEST = {"IADD3", "UIADD3", "LEA", "ULEA", "IMAD", "VOTE"}
PRED_DEST = {"FSETP", "HSETP2", "DSETP", "PLOP3", "UPLOP3", "UISETP"}
_REG = re.compile(r"^-?U?R(\d+|Z)(\.reuse)?$")


def _bump(base: tuple, k: int) -> tuple:
    return tuple((s, c * k & M32) for s, c in base if c * k & M32)


def _merge(a: tuple, b: tuple, sign: int = 1) -> tuple:
    if not b and len(a) < 2 and not (a and not a[0][1]):  # already in merged form
        return a
    d: dict[Any, int] = dict(a)
    for s, c in b:
        d[s] = (d.get(s, 0) + sign * c) & M32
    return tuple(sorted(((s, c) for s, c in d.items() if c), key=repr))


# operand text -> (result without registers or None, register name or None, negated, is a register)
_TOK: dict[str, tuple[Any, str, bool, bool]] = {}
# operand lists -> the source operands, without predicates
_SRC: dict[tuple[str, ...], list[str]] = {}
_PRED = re.compile(r"^!?U?P(T|\d)$")
_NUM = re.compile(r"^(0x[0-9a-f]+|\d+)$")
_BANK = re.compile(r"c\[0x([0-9a-f]+)\]\[0x([0-9a-f]+)\]")
# memory operand text -> its address terms
_ADDR: dict[str, list[str] | None] = {}


def _tok(tok: str) -> tuple[Any, str, bool, bool]:
    t = tok.strip().replace(".reuse", "")
    neg = t.startswith("-")
    t = t.lstrip("-")
    r: Val | None = None
    reg = False
    if t in ("RZ", "URZ"):
        r = ((), ZERO)
    elif _NUM.match(t):
        r = ((), (int(t, 16) if t.startswith("0x") else int(t),) * 32)
    elif t.startswith("c["):
        m = _BANK.match(t)
        r = (((("c", int(m.group(1), 16), int(m.group(2), 16)), 1),), ZERO) if m else None
    elif _REG.match(t):
        reg = True
    if r is not None and neg:
        r = (_bump(r[0], -1), tuple(-o & M32 for o in r[1]))
    _TOK[tok] = out = (r, t, neg, reg)
    return out


def _add(*xs: Val) -> Val:
    b: tuple = ()
    offs = list(ZERO)
    for x in xs:
        b = _merge(b, x[0])
        offs = [(o + y) & M32 for o, y in zip(offs, x[1], strict=True)]
    return (b, tuple(offs))


def _mul(x: Val, y: Val) -> Val | None:
    if Lanes.exact(y) and Lanes.uniform(y):
        k = y[1][0]
        return (_bump(x[0], k), tuple(o * k & M32 for o in x[1]))
    if Lanes.exact(x) and Lanes.uniform(x):
        return _mul(y, x)
    if Lanes.exact(x) and Lanes.exact(y):
        return ((), tuple(p * q & M32 for p, q in zip(x[1], y[1], strict=True)))
    return None


def _shl(x: Val, s: Val) -> Val | None:
    if not (Lanes.exact(s) and Lanes.uniform(s)):
        return None
    return _mul(x, ((), (1 << (s[1][0] & 31),) * 32))


class Lanes:
    """Per-warp lane values; `vals` is the warp's lane-0 value map (sm.Warp.vals), consulted for
    uniform registers the tracker computes no value for."""

    def __init__(
        self, tid0: int, block_dim: tuple[int, int, int], const_bank: dict | None = None
    ) -> None:
        self.const_bank = const_bank or {}
        self.v: dict[str, Val] = {}
        self.unknown: set[str] = set()
        self.tid0 = tid0
        self.bd = block_dim
        self.fresh = 0
        self.p: dict[str, tuple[bool, ...]] = {}  # round 11: predicates that split the lanes
        self.loaded: set[str] = set()  # round 14: registers computed from loaded data
        self.wide: Val | None = None  # round 15: the high half `_eval` just computed, if any
        self.why: dict[str, str] = {}  # round 15: unknown register -> the opcode that broke it
        self.sym_op: dict[int, str] = {}  # round 15: fresh symbol -> the opcode that made it
        self.spill = False  # round 18: values follow the compiler's spills through the stack
        self.stack: dict[str, tuple[Val | None, Val | None]] = {}

    # --- operands ---------------------------------------------------------------------------
    def get(self, tok: str, vals: dict[str, Any]) -> Val | None:
        fixed, t, neg, reg = _TOK.get(tok) or _tok(tok)
        if not reg:
            return fixed
        if t in self.v:
            r: Val = self.v[t]
        elif t in self.unknown:
            return None
        elif t in vals:
            r = ((), (vals[t] & M32,) * 32)
        else:  # never written in this replay: a uniform value the replay does not know
            r = (((("u", t), 1),), ZERO)
        if neg:
            r = (_bump(r[0], -1), tuple(-o & M32 for o in r[1]))
        return r

    @staticmethod
    def uniform(x: Val) -> bool:
        return all(o == x[1][0] for o in x[1])

    @staticmethod
    def exact(x: Val) -> bool:
        return not x[0]

    # --- instructions -------------------------------------------------------------------------
    def step(
        self,
        base: str,
        op: str,
        args: tuple[str, ...],
        vals: dict[str, Any],
        runs: bool | None,
        guard: str | None = None,
    ):
        """Update the destination of one executed instruction. `args`: stripped operands; `runs`:
        the guard predicate (True, False, or None when the replay does not know it); `guard`: its
        token."""
        if base == "ISETP" and runs is not False:
            self._set_pred(op, args, vals)
        elif base in PRED_DEST and runs is not False:  # a predicate the lanes do not follow
            self.p.pop(args[0], None)
            self.p.pop(args[1], None)
        elif base == "LOP3" and args[0][0] == "P" and runs is not False:
            self._lop_pred(args, vals)
        elif base == "R2P":
            self.p.clear()
        elif base in CARRY_DEST and len(args) > 2 and args[1][0] == "P":  # a carry out
            self.p.pop(args[1], None)
            self.p.pop(args[2], None)
        if self.spill and runs is not False and base in ("STL", "LDL"):
            if self._spill(base, op, args, vals):
                return None
        if base in NO_DEST or not args or not _REG.match(args[0]):
            return None
        d = args[0]
        lanes = self.lanes_of(guard) if guard else None
        if lanes is not None:  # round 11: a guard known per lane: the old value on the others
            old, new = self.get(d, vals), self._eval(base, op, args, vals)
            self.wide = None
            mixed = _merge_lanes(lanes, new, old) if isinstance(new, tuple) else None
            if mixed is None:
                self.why[d] = self._cause(op, args)
            self._set(d, mixed, op)
            return None
        if runs is False:
            return None
        if runs is None:  # a guarded write the replay cannot decide: unknown afterwards
            self.why[d] = "guard"
            self._set(d, None, op)
            return None
        out = self._eval(base, op, args, vals)
        if base in ("LDC", "LDCU") and ".64" in op and isinstance(out, tuple) and not out[0]:
            # both halves of a declared 64-bit read (the replay's pairs track the low half)
            m = re.search(r"c\[0x([0-9a-f]+)\]\[0x([0-9a-f]+)\]", ",".join(args[1:]))
            hi = self.const_bank.get((int(m.group(1), 16), int(m.group(2), 16) + 4)) if m else None
            self._set(d, out, op)
            r = re.match(r"^(U?R)(\d+)$", d)
            if hi is not None and r:
                self._set(f"{r.group(1)}{int(r.group(2)) + 1}", ((), (hi & M32,) * 32), "")
            return None
        if out == "uniform":  # the lane-0 value once the warp's value tracking has computed it

            def done(vals_after: dict[str, Any], d: str = d, op: str = op) -> None:
                v = vals_after.get(d)
                if v is None and base in ("S2R", "S2UR"):  # one register, one symbol
                    self._set(d, (((("u", args[1]), 1),), ZERO), op)
                    return
                self._set(d, ((), (v & M32,) * 32) if v is not None else "uniform", op)

            return done
        if out is None:
            self.why[d] = self._cause(op, args)
        self._set(d, out, op)
        if self.wide is not None:
            r = re.match(r"^(U?R)(\d+)$", d)
            if r:
                self._set(f"{r.group(1)}{int(r.group(2)) + 1}", self.wide, "")
            self.wide = None
        if out is None and (base in ("LDG", "BREV") or self.loaded.intersection(args[1:])):
            self.loaded.add(d)
        elif self.loaded:
            self.loaded.discard(d)
        return None

    def _cause(self, op: str, args: tuple[str, ...]) -> str:
        """The opcode that broke a value: a source's own cause, else this instruction."""
        for t in args[1:]:
            c = self.why.get(_TOK.get(t, _tok(t))[1])
            if c and _TOK[t][1] in self.unknown:
                return c
        return ".".join(op.split(".")[:3])

    def address_gap(self, args: str, addr: Val | None) -> tuple[str, str] | None:
        """Round 15, the coverage count: None for a resolved address; else ("unknown", cause), the
        lanes are not known, or ("opaque", cause), the lanes are known but the base is a uniform
        value the replay gave a fresh symbol, so nothing links it to its buffer."""
        if addr is None:
            for t in _ADDR.get(args) or ():
                t = t.lstrip("-")
                if t in self.unknown:
                    return ("unknown", self.why.get(t, "?"))
            return ("unknown", "no address")
        for sym, _ in addr[0]:
            if sym[0] == "u" and len(sym) == 3:
                return ("opaque", self.sym_op.get(sym[2], "?"))
        return None

    def _spill(self, base: str, op: str, args: tuple[str, ...], vals: dict[str, Any]) -> bool:
        """A store to or load from a fixed slot of the frame (`[R1+0x8]`): the compiler's spills.
        True when the load found its value."""
        slot, reg = (args[0], args[1]) if base == "STL" else (args[1], args[0])
        r = re.match(r"^R(\d+)$", reg)
        if not r or not slot.startswith("[R1+"):
            return False
        hi = f"R{int(r.group(1)) + 1}"
        if base == "STL":
            wide = self.get(hi, vals) if ".64" in op else None
            self.stack[slot] = (self.get(reg, vals), wide)
            return False
        got = self.stack.get(slot)
        if got is None or got[0] is None:
            return False
        self.v[reg] = got[0]
        self.unknown.discard(reg)
        self.loaded.discard(reg)
        if ".64" in op and got[1] is not None:
            self.v[hi] = got[1]
            self.unknown.discard(hi)
            self.loaded.discard(hi)
        return True

    def lanes_of(self, tok: str) -> tuple[bool, ...] | None:
        """The 32 lanes of a predicate token (`P1`, `!P1`, `@!P1`), when known."""
        t = tok.lstrip("@")
        v = self.p.get(t.lstrip("!"))
        return v if v is None or not t.startswith("!") else tuple(not x for x in v)

    def _lop_pred(self, a: tuple[str, ...], vals: dict[str, Any]) -> None:
        """A bitwise function as a compare with zero: `LOP3.LUT P0, RZ, a, b, c, lut, !PT`."""
        self.p.pop(a[0], None)
        g = [self.get(t, vals) for t in a[2:6]]
        if a[-1] == "!PT" and len(a) == 7 and None not in g and all(self.exact(x) for x in g):
            v = zip(g[0][1], g[1][1], g[2][1], strict=True)
            self.p[a[0]] = tuple(lut3(x, y, z, g[3][1][0]) != 0 for x, y, z in v)

    def _set_pred(self, op: str, a: tuple[str, ...], vals: dict[str, Any]) -> None:
        """Round 11 (hand_stencil), round 15 as a class: a compare of lane values is known per lane
        when the values are exact or differ by exact lanes on one base; the two destinations are
        the compare and its inverse, each combined with the input predicate; anything else forgets
        them."""
        self.p.pop(a[0], None)
        self.p.pop(a[1], None)
        g = [self.get(t, vals) for t in a[2:4]]
        if len(a) < 5 or any(x is None for x in g) or g[0][0] != g[1][0]:
            return
        out = _compare(op, g[0], g[1])
        tok = a[4]
        pin = (True,) * 32 if tok == "PT" else self.lanes_of(tok)
        if tok == "!PT":
            pin = (False,) * 32
        if out is None or pin is None:
            return
        comb = "AND" if ".AND" in op else "OR" if ".OR" in op else "XOR" if ".XOR" in op else None
        if comb is None:
            return
        fn = {"AND": lambda c, q: c and q, "OR": lambda c, q: c or q, "XOR": lambda c, q: c != q}
        if re.match(r"^P\d$", a[0]):
            self.p[a[0]] = tuple(fn[comb](c, q) for c, q in zip(out, pin, strict=True))
        if re.match(r"^P\d$", a[1]):
            self.p[a[1]] = tuple(fn[comb](not c, q) for c, q in zip(out, pin, strict=True))

    def _set(self, d: str, val: Val | str | None, op: str) -> None:
        hi = None
        if ".64" in op or ".WIDE" in op:
            m = re.match(r"^(U?R)(\d+)$", d)
            hi = f"{m.group(1)}{int(m.group(2)) + 1}" if m else None
        if val == "uniform":
            self.fresh += 1
            self.sym_op[self.fresh] = ".".join(op.split(".")[:3])
            val = (((("u", d, self.fresh), 1),), ZERO)
        if val is None:
            self.v.pop(d, None)
            self.unknown.add(d)
        else:
            self.v[d] = val  # type: ignore[assignment]
            self.unknown.discard(d)
        if hi:  # the high half of a pair: not tracked
            self.v.pop(hi, None)
            self.unknown.add(hi)
            self.why[hi] = "high:" + ".".join(op.split(".")[:3])

    def _eval(
        self, base: str, op: str, a: tuple[str, ...], vals: dict[str, Any]
    ) -> Val | str | None:
        d = a[0]
        if base in ("S2R", "S2UR"):
            sr = a[1] if len(a) > 1 else ""
            lin = tuple(self.tid0 + i for i in LANES)
            bx, by, _ = self.bd
            if "SR_TID.X" in sr:
                return ((), tuple(t % bx for t in lin))
            if "SR_TID.Y" in sr:
                return ((), tuple((t // bx) % by for t in lin))
            if "SR_TID.Z" in sr:
                return ((), tuple(t // (bx * by) for t in lin))
            if "SR_LANEID" in sr:
                return ((), LANES)
            return "uniform"
        if base in ("LDC", "LDCU"):
            m = re.search(r"c\[0x([0-9a-f]+)\]\[0x([0-9a-f]+)\]", ",".join(a[1:]))
            if m and "R" not in a[1].split("[")[-1]:
                c = (int(m.group(1), 16), int(m.group(2), 16))
                if c in self.const_bank:  # an argument the caller declares
                    return ((), (self.const_bank[c] & M32,) * 32)
                return ((("c", *c), 1),), ZERO
            return "uniform"
        if base == "CS2R" and a[-1] == "SRZ":  # the compiler's zero
            return ((), ZERO)
        if base == "CS2R":
            return "uniform"
        if base == "HFMA2" and a[1:3] == ("-RZ", "RZ") and len(a) == 5:
            # the compiler's 32-bit constant: two half-precision immediates
            try:
                return ((), (_half(a[3]) << 16 | _half(a[4]),) * 32)
            except (ValueError, OverflowError):
                return None
        if base in ("I2F", "I2FP", "F2I", "MUFU", "FMUL", "FADD", "FFMA", "POPC", "FLO", "BREV"):
            # the steps of a division by a run-time count: lanes by their bits (round 15)
            gs = [self._float_src(t, base, vals) for t in a[1:] if not _PRED.match(t)]
            if gs and all(x is not None and self.exact(x) for x in gs):
                return _bit_step(base, op, gs)
            return None
        if base in WRITES_UNKNOWN:
            return None
        src = _SRC.get(a)
        if src is None:
            src = _SRC[a] = [t for t in a[1:] if not _PRED.match(t)]
        g = [self.get(t, vals) for t in src]
        if base in ("SEL", "USEL", "VIMNMX", "IMNMX") and len(a) >= 4 and None not in g[:2]:
            r = self._select(base, op, a, g)
            if r is not None:
                return r
        try:
            r = self._linear(base, op, a, src, g, vals)
        except (IndexError, ValueError, TypeError):
            r = None
        if r is not None:
            return r
        if base in ("LOP3", "ULOP3") and len(g) >= 4 and None not in g[:4]:
            return self._lop3(g) or self._opaque(g, d, op)
        if all(x is not None and self.uniform(x) and self.exact(x) for x in g):
            return "uniform"
        if all(x is not None and self.uniform(x) for x in g):
            self.fresh += 1
            self.sym_op[self.fresh] = ".".join(op.split(".")[:3])
            return (((("u", d, self.fresh), 1),), ZERO)
        return None

    def _float_src(self, tok: str, base: str, vals: dict[str, Any]) -> Val | None:
        """A source of a float step: a decimal immediate by its bits, a negated register by its
        sign bit."""
        if base not in FLOAT_STEP:
            return self.get(tok, vals)
        t = tok.lstrip("-")
        if _DEC.match(t):
            x = float(t)
            return ((), (_b(-x if tok[0] == "-" else x),) * 32)
        v = self.get(t, vals)
        if v is not None and tok[0] == "-" and self.exact(v):
            return ((), tuple(o ^ 0x80000000 for o in v[1]))
        return v

    def _opaque(self, g: list[Val | None], d: str, op: str) -> Val | str | None:
        if all(x is not None and self.uniform(x) and self.exact(x) for x in g):
            return "uniform"
        if all(x is not None and self.uniform(x) for x in g):
            self.fresh += 1
            self.sym_op[self.fresh] = ".".join(op.split(".")[:3])
            return (((("u", d, self.fresh), 1),), ZERO)
        return None

    def _select(self, base: str, op: str, a: tuple[str, ...], g: list[Val | None]) -> Val | None:
        """Round 11 (hand_stencil): SEL by a lane predicate; min (PT), max (!PT) of exact lanes."""
        if base in ("SEL", "USEL"):
            lanes = self.lanes_of(a[3])
            if a[3] in ("PT", "UPT"):
                lanes = (True,) * 32
            return _merge_lanes(lanes, g[0], g[1]) if lanes is not None else None
        x, y = g[0], g[1]
        if a[3] not in ("PT", "!PT") or not (self.exact(x) and self.exact(y)):
            return None
        pick = min if a[3] == "PT" else max
        sign = (lambda v: v) if ".U32" in op else _signed
        return ((), tuple(pick(p, q, key=sign) for p, q in zip(x[1], y[1], strict=True)))

    def _high(self, tok: str, vals: dict[str, Any]) -> Val | None:
        """The high half of the pair `tok` starts; a constant's is zero."""
        t = tok.replace(".reuse", "").lstrip("-")
        if t in ("RZ", "URZ") or _NUM.match(t):
            return ((), ZERO)
        r = re.match(r"^(U?R)(\d+)$", t)
        return self.get(f"{r.group(1)}{int(r.group(2)) + 1}", vals) if r else None

    def _linear(self, base, op, a, src, g, vals) -> Val | None:
        if base in ("MOV", "UMOV", "MOV32I", "R2UR") and g and g[0] is not None:
            if ".64" in op:
                self.wide = self._high(src[0], vals)
            return g[0]
        if base == "LOP3" and len(g) >= 4 and g[3] == ((), (0xC0,) * 32):
            # a & b with one of them known to be zero: the other need not be known
            if any(x is not None and x == ((), ZERO) for x in g[:2]):
                return ((), ZERO)
        if op.startswith("IMAD.MOV"):
            return g[-1] if g and g[-1] is not None else None
        if None in g:
            return None
        if base in ("IADD3", "UIADD3", "IADD", "VIADD", "UIADD") or op.startswith("IADD.64"):
            if ".64" in op and all(self.exact(x) for x in g):
                self.wide = _add_64(g, [self._high(t, vals) for t in src])
            return _add(*g)
        if base in ("IMAD", "UIMAD"):
            return self._imad(op, src, g, vals)
        if base in ("LEA", "ULEA"):
            return self._lea(op, src, g)
        if base in ("SHF", "USHF"):
            return _shf(op, src, g)
        if base == "PRMT" and len(g) >= 3 and all(self.exact(x) for x in g[:3]):
            # round 10: bytes of a and c by the selector's nibbles (TileLang's 16-bit products)
            return (
                (),
                tuple(prmt(*v) for v in zip(g[0][1], g[1][1], g[2][1], strict=True)),
            )
        if base == "IABS" and self.exact(g[0]):
            return ((), tuple(abs(_signed(o)) & M32 for o in g[0][1]))
        return None

    def _imad(self, op: str, src: list[str], g: list[Val], vals: dict[str, Any]) -> Val | None:
        if ".HI" not in op and ".WIDE" not in op:
            p = _mul(g[0], g[1])
            return _add(p, g[2]) if p is not None and len(g) > 2 else p
        sign = (lambda v: v) if ".U32" in op else _signed
        if ".HI" in op:  # the high half of a product plus the pair (c+1:c)
            hi = self._high(src[2], vals)
            if hi is None or not all(self.exact(x) for x in (g[0], g[1], g[2], hi)):
                return None
            v = zip(g[0][1], g[1][1], g[2][1], hi[1], strict=True)
            return (
                (),
                tuple(((sign(x) * sign(y) + (h << 32 | z)) >> 32) & M32 for x, y, z, h in v),
            )
        p = _mul(g[0], g[1])  # .WIDE: the low half keeps a base; the high half only exact lanes
        hi = self._high(src[2], vals)
        if self.exact(g[0]) and self.exact(g[1]) and self.exact(g[2]) and hi is not None:
            if self.exact(hi):
                v = zip(g[0][1], g[1][1], g[2][1], hi[1], strict=True)
                wide = [(sign(x) * sign(y) + (h << 32 | z)) for x, y, z, h in v]
                self.wide = ((), tuple((w >> 32) & M32 for w in wide))
        return _add(p, g[2]) if p is not None else None

    def _lea(self, op: str, src: list[str], g: list[Val]) -> Val | None:
        if ".HI" not in op:
            p = _shl(g[0], g[2])
            return _add(p, g[1]) if p is not None else None
        # (hi:a >> 32 - s) + b; the carry of LEA.HI.X into the high half is not followed
        s = g[-1] if ".SX32" in op else g[3]
        if not (self.exact(s) and self.uniform(s)):
            return None
        k = 32 - (s[1][0] & 31)
        if ".SX32" in op:
            q = (
                _shr(((), tuple(_signed(o) & M32 for o in g[0][1])), k)
                if self.exact(g[0])
                else None
            )
            return _add(q, g[1]) if q is not None else None
        if src[2] in ("RZ", "URZ"):
            q = _shr(g[0], k)
            return _add(q, g[1]) if q is not None else None
        if self.exact(g[0]) and self.exact(g[2]):
            v = zip(g[0][1], g[2][1], strict=True)
            return _add(((), tuple(((h << 32 | o) >> k) & M32 for o, h in v)), g[1])
        return None

    def _lop3(self, g: list[Val]) -> Val | None:
        """Round 12 (a hash masked to a table), round 15 as a class: any bitwise function: exact
        lanes by their bits; over one base, by the bits the base leaves free. Above those, a
        function that ignores the base gives exact bits; one that passes it (a mask of all ones,
        or a field or'ed in) keeps it: the compiler or's fields only over bits the base leaves
        clear (round 8, round 14)."""
        if not (self.exact(g[3]) and self.uniform(g[3])):
            return None
        lut = g[3][1][0]
        based = [i for i in range(3) if g[i][0]]
        if not based:
            v = zip(g[0][1], g[1][1], g[2][1], strict=True)
            return ((), tuple(lut3(x, y, z, lut) for x, y, z in v))
        if len(based) > 1:
            return None
        i = based[0]
        free = (1 << _free_bits(g[i][0])) - 1
        keep = M32 & ~free
        low, high = [], []
        passes = same = True
        for n in LANES:
            ops = [g[0][1][n], g[1][1][n], g[2][1][n]]
            low.append(lut3(*ops, lut) & free)
            ops[i] = 0
            f0 = lut3(*ops, lut) & keep
            ops[i] = M32
            f1 = lut3(*ops, lut) & keep
            high.append(f0)
            same = same and f0 == f1
            passes = passes and f1 == keep
        if same:
            return ((), tuple(p | q for p, q in zip(low, high, strict=True)))
        if passes:
            offs = g[i][1]
            return (
                g[i][0],
                tuple(o & keep | q | p for o, p, q in zip(offs, low, high, strict=True)),
            )
        return None

    def pred_mixed(self, base: str, op: str, args: list[str], vals: dict[str, Any]) -> bool | None:
        """Round 9: does a predicate write split the warp's lanes? True: the lanes' values give
        both outcomes, or the value is data the replay does not know (a load's); False: all lanes
        agree; None: not a compare of lane values."""
        if base == "ISETP":
            src, lut = args[2:4], None
        elif base == "LOP3":
            src, lut = args[2:5], self.get(args[5], vals)
        else:
            return None
        g = [self.get(t, vals) for t in src]
        if any(x is None for x in g):
            return True
        if any(not self.exact(x) for x in g):
            return False if all(self.uniform(x) for x in g) else True
        if lut is not None:
            k = lut[1][0]
            out = {lut3(*(x[1][i] for x in g), k) != 0 for i in LANES}
        else:
            r = _compare(op, g[0], g[1])
            if r is None:
                return None
            out = set(r)
        return len(out) > 1

    # --- addresses ----------------------------------------------------------------------------
    def address(self, args: str, vals: dict[str, Any]) -> Val | None:
        """The 32 lanes' address of a memory operand `[R2.64+0x10]` / `desc[UR4][R2.64+UR6]`."""
        if args in _ADDR:
            terms = _ADDR[args]
        else:
            m = re.findall(r"\[([^\]]*)\]", args)
            terms = _ADDR[args] = (
                [
                    t.replace(".64", "").replace(".U32", "").replace(".X4", "")
                    for t in m[-1].split("+")
                    if t
                ]
                if m
                else None
            )
        if terms is None:
            return None
        out: Val = ((), ZERO)
        for t in terms:
            v = self.get(t, vals)
            if v is None:
                return None
            out = (
                _merge(out[0], v[0]),
                tuple((o + y) & M32 for o, y in zip(out[1], v[1], strict=True)),
            )
        return out


def _half(tok: str) -> int:
    """The bits of a half-precision immediate as nvdisasm prints it."""
    return struct.unpack("<H", struct.pack("<e", float(tok)))[0]


def _free_bits(base: tuple) -> int:
    """Low bits of a value the base leaves zero: a product's by its coefficient, a pointer's by its
    alignment."""
    bits = 32
    for sym, k in base:
        bits = min(bits, (k & -k).bit_length() - 1 + (7 if sym[0] == "c" or len(sym) == 2 else 0))
    return min(bits, 31)


def _shr(x: Val, k: int) -> Val | None:
    """x >> k; a base only if its bits below k are zero."""
    if any(c & ((1 << k) - 1) for _, c in x[0]):
        return None
    base = tuple((s, c >> k) for s, c in x[0])
    return (base, tuple(o >> k for o in x[1]))


def _shf(op: str, src: list[str], g: list[Val]) -> Val | None:
    """SHF over a pair (hi:lo) in the forms compiled code uses; lanes' own shift counts allowed."""
    lo, s, hi = g[0], g[1], g[2]
    left = ".L." in op
    if left and ".HI" not in op:
        if s[0]:
            return None
        if Lanes.uniform(s):
            return _shl(lo, s)
        if not lo[0]:
            return ((), tuple((o << (c & 31)) & M32 for o, c in zip(lo[1], s[1], strict=True)))
        return None
    if s[0]:
        return None
    if not left and Lanes.uniform(s) and src[2] in ("RZ", "URZ") and ".HI" not in op:
        return _shr(lo, s[1][0] & 31)
    if lo[0] or hi[0]:
        if not left and Lanes.uniform(s) and src[0] in ("RZ", "URZ") and ".U32" in op:
            return _shr(hi, s[1][0] & 31)
        return None
    out = []
    for a, c, b in zip(lo[1], s[1], hi[1], strict=True):
        k = c & (63 if "64" in op else 31)
        if left:  # SHF.L.U64.HI
            out.append(((b << 32 | a) << k >> 32) & M32)
        elif ".S32" in op:
            out.append((_signed(b) >> k) & M32)
        elif ".HI" in op:
            out.append((b >> k) & M32)
        else:
            out.append(((b << 32 | a) >> k) & M32)
    return ((), tuple(out))


def _add_64(g: list[Val], hi: list[Val | None]) -> Val | None:
    """The high half of a 64-bit sum of exact lanes, when every high half is known."""
    if any(h is None or h[0] for h in hi):
        return None
    out = []
    for n in LANES:
        t = 0
        for x, h in zip(g, hi, strict=True):
            t += x[1][n] | h[1][n] << 32
        out.append((t >> 32) & M32)
    return ((), tuple(out))


def _f(v: int) -> float:
    return struct.unpack("<f", struct.pack("<I", v & M32))[0]


def _b(x: float, up: bool = False) -> int:
    try:
        r = struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:
        return 0x7F800000 if x > 0 else 0xFF800000
    if up and _f(r) < x:
        r += 1
    return r


def _bit_step(base: str, op: str, g: list[Val]) -> Val | None:
    """Exact lanes through a bit count, a conversion or a float step: the division by a run-time
    count the compiler writes, whose integer tail corrects a last-place difference."""

    def one(*v: int) -> int | None:
        x = v[0]
        if base == "POPC":
            return bin(x).count("1")
        if base == "BREV":
            return int(f"{x:032b}"[::-1], 2)
        if base == "FLO":
            return x.bit_length() - 1 & M32 if x else M32
        if base in ("I2F", "I2FP"):
            n = _signed(x) if ".S32" in op or (base == "I2F" and ".U32" not in op) else x
            return _b(float(n), ".RP" in op)
        if base == "F2I":
            f = _f(x)
            return 0 if not math.isfinite(f) else max(0, int(f)) & M32
        if base == "MUFU" and ".RCP" in op and _f(x):
            return _b(1.0 / _f(x))
        if base == "FMUL":
            return _b(_f(x) * _f(v[1]))
        if base == "FADD":
            return _b(_f(x) + _f(v[1]))
        if base == "FFMA":
            return _b(_f(x) * _f(v[1]) + _f(v[2]))
        return None

    if base in ("I2F", "F2I") and not any(p in op for p in ("U32", "S32", "F32")):
        if op.split(".")[1:2] in (["U16"], ["S16"], ["F16"], ["U64"], ["S64"], ["F64"]):
            return None
    out = [one(*v) for v in zip(*(x[1] for x in g), strict=True)]
    return None if None in out else ((), tuple(out))  # type: ignore[arg-type]


def _signed(v: int) -> int:
    return v - (1 << 32) if v >= 1 << 31 else v


def _merge_lanes(lanes: tuple[bool, ...], a: Val | None, b: Val | None) -> Val | None:
    """`a` on the lanes set, `b` on the others; one value when only one side is ever taken."""
    if all(lanes):
        return a
    if not any(lanes):
        return b
    if a is None or b is None:
        return None
    if a[0] != b[0]:  # round 15 (`bl15_twobuf`): two buffers: far apart under one base
        if not a[0] or not b[0]:
            return None
        return (
            a[0],
            tuple(x if p else y + APART for p, x, y in zip(lanes, a[1], b[1], strict=True)),
        )
    return (a[0], tuple(x if p else y for p, x, y in zip(lanes, a[1], b[1], strict=True)))


def _compare(op: str, a: Val, b: Val) -> tuple[bool, ...] | None:
    m = re.match(r"ISETP\.(\w+)(\.U32|\.U64)?", op)
    cmp = {
        "GT": lambda x, y: x > y,
        "GE": lambda x, y: x >= y,
        "LT": lambda x, y: x < y,
        "LE": lambda x, y: x <= y,
        "NE": lambda x, y: x != y,
        "EQ": lambda x, y: x == y,
    }.get(m[1] if m else "")
    if cmp is None:
        return None
    sign = (lambda v: v) if m[2] else _signed
    return tuple(cmp(sign(p), sign(q)) for p, q in zip(a[1], b[1], strict=True))


def prmt(a: int, sel: int, c: int) -> int:
    """Bytes of a (0-3) and c (4-7) by the selector's nibbles; a nibble's top bit replicates that
    byte's sign (round 13: 0x9910, a 16-bit sign extension in the compiler's division by 3)."""
    v = 0
    for i in range(4):
        n = (sel >> (4 * i)) & 15
        b = ((c if n & 4 else a) >> (8 * (n & 3))) & 255
        v |= ((0xFF if b & 128 else 0) if n & 8 else b) << (8 * i)
    return v


def lut3(a: int, b: int, c: int, lut: int) -> int:
    """The bitwise three-input function `lut` of the LOP3 family."""
    v = 0
    for i in range(8):
        if (lut >> i) & 1:
            v |= (a if i & 4 else ~a) & (b if i & 2 else ~b) & (c if i & 1 else ~c)
    return v & M32


def footprint(addr: Val, width: int) -> tuple[dict[tuple, int], int]:
    """(line (base, line index) -> mask of its 32-byte sectors, sectors) the 32 lanes touch with
    `width`-byte accesses."""
    lines: dict[tuple, int] = {}
    sectors = set()
    for o in addr[1]:
        for s in range(o // 32, (o + width - 1) // 32 + 1):
            sectors.add(s)
            k = (addr[0], s // 4)
            lines[k] = lines.get(k, 0) | 1 << (s % 4)
    return lines, len(sectors)


def contiguous(addr: Val, width: int) -> bool:
    """Round 11: the 32 lanes follow one another, `width` bytes apart."""
    return all(b - a == width for a, b in zip(addr[1], addr[1][1:], strict=False))


def shared_part_sector(addr: Val, width: int) -> bool:
    """Round 8: some 32-byte sector holds more than one lane and is not full."""
    lanes: dict[int, int] = {}
    for o in addr[1]:
        lanes[o >> 5] = lanes.get(o >> 5, 0) + 1
    return any(1 < n and n * width < 32 for n in lanes.values())


def spread_lines(lines: dict[tuple, int]) -> bool:
    """Round 12: the lines an instruction touches are not one contiguous run."""
    idx = [k[1] for k in lines]
    return max(idx) - min(idx) >= len(idx)


def width_of(op: str) -> int:
    parts = op.split(".")
    if "256" in parts:
        return 32
    if "128" in parts:
        return 16
    if "64" in parts:
        return 8
    if any(p in parts for p in ("U16", "S16")):
        return 2
    if any(p in parts for p in ("U8", "S8")):
        return 1
    return 4

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
"""

from __future__ import annotations

import re
from typing import Any

Val = tuple[tuple, tuple]  # (base, offsets[32])
M32 = 0xFFFFFFFF
LANES = tuple(range(32))
ZERO = (0,) * 32
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
        if base in NO_DEST or not args or not _REG.match(args[0]):
            return None
        d = args[0]
        lanes = self.lanes_of(guard) if guard else None
        if lanes is not None:  # round 11: a guard known per lane: the old value on the others
            old, new = self.get(d, vals), self._eval(base, op, args, vals)
            self._set(d, _merge_lanes(lanes, new, old) if isinstance(new, tuple) else None, op)
            return None
        if runs is False:
            return None
        if runs is None:  # a guarded write the replay cannot decide: unknown afterwards
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
                self._set(d, ((), (v & M32,) * 32) if v is not None else "uniform", op)

            return done
        self._set(d, out, op)
        return None

    def lanes_of(self, tok: str) -> tuple[bool, ...] | None:
        """The 32 lanes of a predicate token (`P1`, `!P1`, `@!P1`), when known."""
        t = tok.lstrip("@")
        v = self.p.get(t.lstrip("!"))
        return v if v is None or not t.startswith("!") else tuple(not x for x in v)

    def _set_pred(self, op: str, a: tuple[str, ...], vals: dict[str, Any]) -> None:
        """Round 11 (hand_stencil): a compare of exact lane values, ANDed with PT, is known per
        lane; anything else forgets the predicate."""
        self.p.pop(a[0], None)
        self.p.pop(a[1], None)
        g = [self.get(t, vals) for t in a[2:4]]
        if len(a) < 5 or a[4] != "PT" or ".AND" not in op or any(x is None for x in g):
            return
        if not all(self.exact(x) for x in g):
            return
        out = _compare(op, g[0], g[1])
        if out is not None and re.match(r"^P\d$", a[0]):
            self.p[a[0]] = out

    def _set(self, d: str, val: Val | str | None, op: str) -> None:
        hi = None
        if ".64" in op or ".WIDE" in op:
            m = re.match(r"^(U?R)(\d+)$", d)
            hi = f"{m.group(1)}{int(m.group(2)) + 1}" if m else None
        if val == "uniform":
            self.fresh += 1
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
        if base == "HFMA2" and a[1:] == ("-RZ", "RZ", "0", "0"):  # the compiler's zero
            return ((), ZERO)
        if base in ("I2F", "F2I", "MUFU"):
            # the steps of a division by a run-time count: a value for uniform operands
            g0 = self.get(a[-1], vals)
            return "uniform" if g0 is not None and self.uniform(g0) and self.exact(g0) else None
        if base in WRITES_UNKNOWN:
            return None
        src = _SRC.get(a)
        if src is None:
            src = _SRC[a] = [t for t in a[1:] if not _PRED.match(t)]
        g = [self.get(t, vals) for t in src]
        if base in ("SEL", "VIMNMX", "IMNMX") and len(a) >= 4 and None not in g[:2]:
            r = self._select(base, op, a, g)
            if r is not None:
                return r
        try:
            r = self._linear(base, op, a, src, g, vals)
        except (IndexError, ValueError, TypeError):
            r = None
        if r is not None:
            return r
        if all(x is not None and self.uniform(x) and self.exact(x) for x in g):
            return "uniform"
        if all(x is not None and self.uniform(x) for x in g):
            self.fresh += 1
            return (((("u", d, self.fresh), 1),), ZERO)
        return None

    def _select(self, base: str, op: str, a: tuple[str, ...], g: list[Val | None]) -> Val | None:
        """Round 11 (hand_stencil): SEL by a lane predicate; min (PT), max (!PT) of exact lanes."""
        if base == "SEL":
            lanes = self.lanes_of(a[3])
            return _merge_lanes(lanes, g[0], g[1]) if lanes is not None else None
        x, y = g[0], g[1]
        if a[3] not in ("PT", "!PT") or not (self.exact(x) and self.exact(y)):
            return None
        pick = min if a[3] == "PT" else max
        sign = (lambda v: v) if ".U32" in op else _signed
        return ((), tuple(pick(p, q, key=sign) for p, q in zip(x[1], y[1], strict=True)))

    def _linear(self, base, op, a, src, g, vals) -> Val | None:
        if base in ("MOV", "UMOV", "MOV32I", "R2UR") and g and g[0] is not None:
            return g[0]
        if base in ("LOP3", "ULOP3") and len(g) >= 4 and None not in g[:4]:
            # round 12: any function of exact lanes (a hash masked to a table)
            if all(self.exact(x) for x in g[:4]) and self.uniform(g[3]):
                lut = g[3][1][0]
                vals3 = zip(g[0][1], g[1][1], g[2][1], strict=True)
                return ((), tuple(lut3(p, q, r, lut) for p, q, r in vals3))
        if base == "LOP3" and len(g) >= 4 and g[3] == ((), (0xC0,) * 32):
            # a & b with one of them known to be zero: the other need not be known
            if any(x is not None and x == ((), ZERO) for x in g[:2]):
                return ((), ZERO)
        if op.startswith("IMAD.MOV"):
            return g[-1] if g and g[-1] is not None else None
        if None in g:
            return None
        if base in ("IADD3", "UIADD3", "IADD", "VIADD", "UIADD") or op.startswith("IADD.64"):
            return _add(*g)
        if base in ("IMAD", "UIMAD") and ".HI" not in op:
            p = _mul(g[0], g[1])
            return _add(p, g[2]) if p is not None and len(g) > 2 else p
        if base in ("LEA", "ULEA") and ".HI" not in op:
            p = _shl(g[0], g[2])
            return _add(p, g[1]) if p is not None else None
        if base == "LEA" and op.startswith("LEA.HI") and ".X" not in op and len(g) >= 4:
            # round 14 (gemv14_*): (RZ:a >> 32 - s) + b, an index divided by a power of two
            if src[2] in ("RZ", "URZ") and self.exact(g[0]) and self.exact(g[3]):
                k = 32 - (g[3][1][0] & 31)
                return _add(((), tuple(o >> k for o in g[0][1])), g[1])
            return None
        if base in ("SHF", "USHF"):
            lo, s, hi = g[0], g[1], g[2]
            if ".L." in op and ".HI" not in op:
                return _shl(lo, s)
            if ".R." in op and self.exact(s) and self.uniform(s):
                k = s[1][0] & 31
                if ".HI" in op and self.exact(hi):
                    return ((), tuple(o >> k for o in hi[1]))
                if ".HI" not in op and self.exact(lo) and src[2] in ("RZ", "URZ"):
                    return ((), tuple(o >> k for o in lo[1]))
            return None
        if base == "PRMT" and len(g) >= 3 and all(self.exact(x) for x in g[:3]):
            # round 10: bytes of a and c by the selector's nibbles (TileLang's 16-bit products)
            if not self.uniform(g[1]):
                return None
            sel = g[1][1][0]
            return ((), tuple(prmt(x, sel, y) for x, y in zip(g[0][1], g[2][1], strict=True)))
        if base in ("LOP3", "ULOP3") and len(g) >= 4:
            # round 8: a | (b & c) over a base: the fields of an address, the base's bits apart
            keep: tuple = ()
            if g[3] == ((), (0xF8,) * 32) and g[0][0] and self.exact(g[1]) and self.exact(g[2]):
                keep, g = g[0][0], [((), g[0][1]), *g[1:]]
            # round 14 (gemv14_*): a | b, a a product of the block index, b below its low zero bits
            if g[3] == ((), (0xFC,) * 32) and g[2] == ((), ZERO) and bool(g[0][0]) != bool(g[1][0]):
                a, b = (g[0], g[1]) if g[0][0] else (g[1], g[0])
                low = min(k & -k for _, k in a[0]) - 1
                if self.exact(b) and all(o & low == 0 for o in a[1]):
                    if all(o & ~low == 0 for o in b[1]):
                        return _add(a, b)
            if not all(self.exact(x) for x in g[:4]) or not self.uniform(g[3]):
                return None
            lut = g[3][1][0]
            out = []
            for x, y, z in zip(g[0][1], g[1][1], g[2][1], strict=True):
                v = 0
                for bit in range(32):
                    i = (((x >> bit) & 1) << 2) | (((y >> bit) & 1) << 1) | ((z >> bit) & 1)
                    v |= ((lut >> i) & 1) << bit
                out.append(v)
            return (keep, tuple(out))
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


def _signed(v: int) -> int:
    return v - (1 << 32) if v >= 1 << 31 else v


def _merge_lanes(lanes: tuple[bool, ...], a: Val | None, b: Val | None) -> Val | None:
    """`a` on the lanes set, `b` on the others; one value when only one side is ever taken."""
    if all(lanes):
        return a
    if not any(lanes):
        return b
    if a is None or b is None or a[0] != b[0]:
        return None
    return (a[0], tuple(x if p else y for p, x, y in zip(lanes, a[1], b[1], strict=True)))


def _compare(op: str, a: Val, b: Val) -> tuple[bool, ...] | None:
    m = re.match(r"ISETP\.(\w+)(\.U32)?", op)
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

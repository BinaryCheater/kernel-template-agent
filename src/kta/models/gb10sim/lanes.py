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
    "ATOMS",
    "ATOMG",
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
    "PRMT",
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
    d: dict[Any, int] = dict(a)
    for s, c in b:
        d[s] = (d.get(s, 0) + sign * c) & M32
    return tuple(sorted(((s, c) for s, c in d.items() if c), key=repr))


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

    # --- operands ---------------------------------------------------------------------------
    def get(self, tok: str, vals: dict[str, Any]) -> Val | None:
        t = tok.strip().replace(".reuse", "")
        neg = t.startswith("-")
        t = t.lstrip("-")
        if t in ("RZ", "URZ"):
            r: Val | None = ((), ZERO)
        elif re.match(r"^(0x[0-9a-f]+|\d+)$", t):
            r = ((), (int(t, 16) if t.startswith("0x") else int(t),) * 32)
        elif t.startswith("c["):
            m = re.match(r"c\[0x([0-9a-f]+)\]\[0x([0-9a-f]+)\]", t)
            r = (((("c", int(m.group(1), 16), int(m.group(2), 16)), 1),), ZERO) if m else None
        elif _REG.match(t):
            if t in self.v:
                r = self.v[t]
            elif t in self.unknown:
                return None
            elif t in vals:
                r = ((), (vals[t] & M32,) * 32)
            else:  # never written in this replay: a uniform value the replay does not know
                r = (((("u", t), 1),), ZERO)
        else:
            return None
        if r is not None and neg:
            r = (_bump(r[0], -1), tuple(-o & M32 for o in r[1]))
        return r

    @staticmethod
    def uniform(x: Val) -> bool:
        return all(o == x[1][0] for o in x[1])

    @staticmethod
    def exact(x: Val) -> bool:
        return not x[0]

    # --- instructions -------------------------------------------------------------------------
    def step(self, base: str, op: str, args: list[str], vals: dict[str, Any], runs: bool | None):
        """Update the destination of one executed instruction. `runs`: the guard predicate
        (True, False, or None when the replay does not know it)."""
        if base in NO_DEST or not args or not _REG.match(args[0].strip()):
            return None
        d = args[0].strip()
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

    def _eval(self, base: str, op: str, args: list[str], vals: dict[str, Any]) -> Val | str | None:
        a = [x.strip() for x in args]
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
        if base in WRITES_UNKNOWN:
            return None
        src = [t for t in a[1:] if not re.match(r"^!?U?P(T|\d)$", t)]
        g = [self.get(t, vals) for t in src]
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

    def _linear(self, base, op, a, src, g, vals) -> Val | None:
        def add(*xs: Val) -> Val:
            b: tuple = ()
            offs = list(ZERO)
            for x in xs:
                b = _merge(b, x[0])
                offs = [(o + y) & M32 for o, y in zip(offs, x[1], strict=True)]
            return (b, tuple(offs))

        def mul(x: Val, y: Val) -> Val | None:
            if self.exact(y) and self.uniform(y):
                k = y[1][0]
                return (_bump(x[0], k), tuple(o * k & M32 for o in x[1]))
            if self.exact(x) and self.uniform(x):
                return mul(y, x)
            if self.exact(x) and self.exact(y):
                return ((), tuple(p * q & M32 for p, q in zip(x[1], y[1], strict=True)))
            return None

        def shl(x: Val, s: Val) -> Val | None:
            if not (self.exact(s) and self.uniform(s)):
                return None
            return mul(x, ((), (1 << (s[1][0] & 31),) * 32))

        if base in ("MOV", "UMOV", "MOV32I", "R2UR") and g and g[0] is not None:
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
            return add(*g)
        if base in ("IMAD", "UIMAD") and ".HI" not in op:
            p = mul(g[0], g[1])
            return add(p, g[2]) if p is not None and len(g) > 2 else p
        if base in ("LEA", "ULEA") and ".HI" not in op:
            p = shl(g[0], g[2])
            return add(p, g[1]) if p is not None else None
        if base in ("SHF", "USHF"):
            lo, s, hi = g[0], g[1], g[2]
            if ".L." in op and ".HI" not in op:
                return shl(lo, s)
            if ".R." in op and self.exact(s) and self.uniform(s):
                k = s[1][0] & 31
                if ".HI" in op and self.exact(hi):
                    return ((), tuple(o >> k for o in hi[1]))
                if ".HI" not in op and self.exact(lo) and src[2] in ("RZ", "URZ"):
                    return ((), tuple(o >> k for o in lo[1]))
            return None
        if base in ("LOP3", "ULOP3") and len(g) >= 4:
            # round 8: a | (b & c) over a base: the fields of an address, the base's bits apart
            keep: tuple = ()
            if g[3] == ((), (0xF8,) * 32) and g[0][0] and self.exact(g[1]) and self.exact(g[2]):
                keep, g = g[0][0], [((), g[0][1]), *g[1:]]
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
            m = re.match(r"ISETP\.(\w+)(\.U32)?", op)
            if not m:
                return None
            cmp = {
                "GT": lambda a, b: a > b,
                "GE": lambda a, b: a >= b,
                "LT": lambda a, b: a < b,
                "LE": lambda a, b: a <= b,
                "NE": lambda a, b: a != b,
                "EQ": lambda a, b: a == b,
            }.get(m[1])
            if cmp is None:
                return None
            sign = (lambda v: v) if m[2] else (lambda v: v - (1 << 32) if v >= 1 << 31 else v)
            out = {cmp(sign(g[0][1][i]), sign(g[1][1][i])) for i in LANES}
        return len(out) > 1

    # --- addresses ----------------------------------------------------------------------------
    def address(self, args: str, vals: dict[str, Any]) -> Val | None:
        """The 32 lanes' address of a memory operand `[R2.64+0x10]` / `desc[UR4][R2.64+UR6]`."""
        m = re.findall(r"\[([^\]]*)\]", args)
        if not m:
            return None
        terms = [t for t in m[-1].split("+") if t]
        out: Val = ((), ZERO)
        for t in terms:
            v = self.get(t.replace(".64", "").replace(".U32", "").replace(".X4", ""), vals)
            if v is None:
                return None
            out = (
                _merge(out[0], v[0]),
                tuple((o + y) & M32 for o, y in zip(out[1], v[1], strict=True)),
            )
        return out


def lut3(a: int, b: int, c: int, lut: int) -> int:
    """The bitwise three-input function `lut` of the LOP3 family."""
    v = 0
    for bit in range(32):
        i = (((a >> bit) & 1) << 2) | (((b >> bit) & 1) << 1) | ((c >> bit) & 1)
        v |= ((lut >> i) & 1) << bit
    return v


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


def shared_part_sector(addr: Val, width: int) -> bool:
    """Round 8: some 32-byte sector holds more than one lane and is not full."""
    lanes: dict[int, int] = {}
    for o in addr[1]:
        lanes[o >> 5] = lanes.get(o >> 5, 0) + 1
    return any(1 < n and n * width < 32 for n in lanes.values())


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

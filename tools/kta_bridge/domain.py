"""The pilot's frozen decision domains and their ranking by the decision-level form.

    python -m tools.kta_bridge.domain sweep CONSTANTS OUT.json    # what to compile
    python -m tools.kta_bridge.domain select CONSTANTS BUILD_DIR OUT.json  # measured subset

Registers are estimated before compilation (`regs_estimate`) and read from compile records after.
"""

import itertools
import json
import random
import sys
from pathlib import Path

from tools.kta_bridge import forms

HGEMM_SHAPE = {"M": 2816, "N": 1760, "K": 2112}
FA_SHAPE = {"B": 1, "S": 1920, "H": 9, "D": 112}
SEED = 20261106
TOP = 8
RANDOM = 16
SWEEP_TOP = 150
SWEEP_RANDOM = 100


def hgemm_domain() -> list[dict]:
    out = []
    for bm, bn, bk, s, t in itertools.product(
        (64, 96, 128, 192, 256),
        (64, 96, 128, 192, 256),
        (16, 32, 48, 64, 96),
        (1, 2, 3, 4),
        (64, 128, 256),
    ):
        x = {"BM": bm, "BN": bn, "BK": bk, "s": s, "T": t}
        if smem_of("hgemm", x) + 1024 <= 102400:
            out.append(x)
    return out


def fa_domain() -> list[dict]:
    out = []
    for bm, bn, s, t in itertools.product(
        (32, 48, 64, 96, 128), (32, 48, 64, 96, 128), (1, 2, 3), (64, 128, 256)
    ):
        x = {"BM": bm, "BN": bn, "s": s, "T": t}
        if smem_of("fa_forward", x) + 1024 <= 102400:
            out.append(x)
    return out


def smem_of(family: str, x: dict) -> int:
    if family == "hgemm":
        return x["s"] * (x["BM"] + x["BN"]) * x["BK"] * 2
    d = FA_SHAPE["D"]
    return x["BM"] * d * 2 + 2 * x["s"] * x["BN"] * d * 2


def regs_estimate(family: str, x: dict) -> int:
    """Fragments on registers plus a constant, from earlier compile records."""
    if family == "hgemm":
        return min(255, int(x["BM"] * x["BN"] / x["T"] + 62))
    d = FA_SHAPE["D"]
    r = -61.5 + 4.09 * x["BM"] * x["BN"] / x["T"] + 1.87 * x["BM"] * d / x["T"]
    r += 87.5 * (x["s"] >= 2) + 45.9 * (x["s"] >= 3)
    return min(255, int(r))


def name_of(family: str, x: dict, cache: str, prefix: str = "lp1") -> str:
    if family == "hgemm":
        tile = f"h_c{x['BM']}x{x['BN']}x{x['BK']}"
    else:
        tile = f"f_c{x['BM']}x{x['BN']}"
    return f"{prefix}{tile}_s{x['s']}_t{x['T']}_{cache}"


def predicted(family: str, x: dict, regs: int, cache: str, c: dict) -> float:
    build = forms.hgemm if family == "hgemm" else forms.fa
    shape = HGEMM_SHAPE if family == "hgemm" else FA_SHAPE
    return forms.time_us(build(shape, x, regs, smem_of(family, x), cache), c[family])


def sweep(c: dict) -> list[dict]:
    """What the compile sweep builds: all of FA, HGEMM's model top and a seeded random remainder."""
    rng = random.Random(SEED)
    ranked = sorted(
        hgemm_domain(),
        key=lambda x: (predicted("hgemm", x, regs_estimate("hgemm", x), "warm", c), str(x)),
    )
    picks = ranked[:SWEEP_TOP] + rng.sample(ranked[SWEEP_TOP:], SWEEP_RANDOM)
    rows = [{"family": "hgemm", "x": x, "cache": "warm"} for x in picks]
    rows += [{"family": "fa_forward", "x": x, "cache": "warm"} for x in fa_domain()]
    for r in rows:
        r["name"] = name_of(r["family"], r["x"], "warm", "lpc")
    return rows


def compiled_regs(build: Path, family: str, x: dict) -> int | None:
    from tools.kta_bridge.dataset import res_usage

    return res_usage(build / name_of(family, x, "warm", "lpc") / "res-usage.txt").get("regs")


def select(c: dict, build: Path) -> list[dict]:
    """Top before and after compile records, plus a seeded random sample."""
    rng = random.Random(SEED)
    chosen = []
    for family, domain in (("hgemm", hgemm_domain()), ("fa_forward", fa_domain())):
        rows = []
        for x in domain:
            regs = compiled_regs(build, family, x)
            if regs is None:  # generator or compiler refused it
                continue
            rows.append(
                {
                    "family": family,
                    "x": x,
                    "regs_estimate": regs_estimate(family, x),
                    "regs": regs,
                    "pre_us": predicted(family, x, regs_estimate(family, x), "warm", c),
                    "post_us": predicted(family, x, regs, "warm", c),
                }
            )
        pre = sorted(rows, key=lambda r: (r["pre_us"], str(r["x"])))
        post = sorted(rows, key=lambda r: (r["post_us"], str(r["x"])))
        roles: dict[str, set[str]] = {}
        for key, ranked in (("pre", pre), ("post", post)):
            for r in ranked[:TOP]:
                roles.setdefault(str(r["x"]), set()).add(key)
        rest = [r for r in rows if str(r["x"]) not in roles]
        for r in rng.sample(rest, min(RANDOM, len(rest))):
            roles.setdefault(str(r["x"]), set()).add("random")
        for r in rows:
            if str(r["x"]) in roles:
                for cache in ("warm", "rotating"):
                    chosen.append(
                        {
                            **r,
                            "cache": cache,
                            "name": name_of(family, r["x"], cache),
                            "roles": sorted(roles[str(r["x"])]),
                            "pre_us": predicted(family, r["x"], r["regs_estimate"], cache, c),
                            "post_us": predicted(family, r["x"], r["regs"], cache, c),
                        }
                    )
    return chosen


def main() -> None:
    c = json.loads(Path(sys.argv[2]).read_text())
    if sys.argv[1] == "sweep":
        rows = sweep(c)
        Path(sys.argv[3]).write_text(json.dumps(rows, indent=0))
    else:
        rows = select(c, Path(sys.argv[3]))
        Path(sys.argv[4]).write_text(json.dumps(rows, indent=0))
    sys.stdout.write(f"{len(rows)} rows\n")


if __name__ == "__main__":
    main()

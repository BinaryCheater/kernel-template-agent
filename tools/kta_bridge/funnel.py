"""The search funnel: MIP shortlist, compile, re-rank on compile records, measure a few.

    python -m tools.kta_bridge.funnel shortlist CONSTANTS OUT.json
    python -m tools.kta_bridge.funnel rerank CONSTANTS SHORTLIST.json BUILD_DIR OUT.json

Shapes here are new to every earlier batch. Controls are drawn before any time is seen.
"""

import json
import random
import re
import sys
import time
from pathlib import Path

from tools.kta_bridge import domain, families, forms, mip
from tools.kta_bridge.dataset import res_usage

SHAPES = {
    "hgemm": {"M": 3200, "N": 2240, "K": 1920},
    "fa_forward": {"B": 1, "S": 2240, "H": 8, "D": 96},
    "moe_nvfp4": {"E": 4, "T": 3840, "N": 2048, "K": 1536},
}
SHORTLIST = {"hgemm": 16, "fa_forward": 12, "moe_nvfp4": 16}
ANCHORS = {
    "hgemm": [{"BM": 128, "BN": 128, "BK": 32, "s": 3, "T": 128}],
    "fa_forward": [{"BM": 64, "BN": 64, "s": 2, "T": 128}],
    "moe_nvfp4": [
        {"BM": 96, "BN": 128, "BK": 32, "s": 3, "T": 128},
        {"BM": 160, "BN": 128, "BK": 32, "s": 3, "T": 128},
    ],
}
RANDOM = 12
MEASURE = 8
SEED = 20261107


def family(name: str) -> mip.Family:
    domain.HGEMM_SHAPE = SHAPES["hgemm"]
    domain.FA_SHAPE = SHAPES["fa_forward"]
    families.MOE_SHAPE = SHAPES["moe_nvfp4"]
    return {"hgemm": families.hgemm, "fa_forward": families.fa, "moe_nvfp4": families.moe_nvfp4}[
        name
    ]("warm")


def name_of(fam: str, x: dict, prefix: str) -> str:
    tile = "x".join(str(x[k]) for k in ("BM", "BN", "BK") if k in x)
    return f"{prefix}{fam[0]}_c{tile}_s{x['s']}_t{x['T']}_warm"


def row(fam: str, x: dict, role: str, prefix: str, c: dict, seconds: float = 0.0) -> dict:
    f = family(fam)
    return {
        "family": fam,
        "shape": SHAPES[fam],
        "x": x,
        "cache": "warm",
        "name": name_of(fam, x, prefix),
        "roles": [role],
        "pre_us": forms.time_us(f.amounts(*key_of(fam, x)), c[fam]),
        "solve_s": seconds,
    }


def key_of(fam: str, x: dict) -> tuple:
    b = x["s"] if fam == "fa_forward" else (x["BK"], x["s"])
    return (x["BM"], x["BN"]), b, x["T"]


def shortlist(c: dict) -> list[dict]:
    rng = random.Random(SEED)
    rows = []
    for fam, k in SHORTLIST.items():
        f = family(fam)
        t0 = time.perf_counter()
        s = mip.solve(f, c[fam], k=k, time_limit=120, diverse=True)
        spent = time.perf_counter() - t0
        picks = [p["x"] for p in s["picks"]]
        pop = [f.assign(*key) for _, key in mip.enumerate_all(f, c[fam])]
        sys.stdout.write(f"{fam}: domain {len(pop)}, {spent:.0f} s, {s['columns']} columns\n")
        rows += [row(fam, x, "mip", "lq", c, spent / k) for x in picks]
        rows += [row(fam, x, "anchor", "lq", c) for x in ANCHORS[fam]]
        rest = [x for x in pop if x not in picks and x not in ANCHORS[fam]]
        # three times the controls needed: the generator refuses some
        rows += [row(fam, x, "random", "lq", c) for x in rng.sample(rest, RANDOM * 3)]
    return rows


def second(c: dict, measured: list[dict]) -> list[dict]:
    """Round two: constants refit on what round one measured, those assignments cut out."""
    rows = []
    for fam, k in SHORTLIST.items():
        f = family(fam)
        seen = tuple(key_of(fam, r["x"]) for r in measured if r["family"] == fam)
        t0 = time.perf_counter()
        s = mip.solve(f, c[fam], k=k, time_limit=120, diverse=True, exclude=seen)
        spent = time.perf_counter() - t0
        sys.stdout.write(f"{fam}: {spent:.0f} s\n")
        rows += [row(fam, p["x"], "mip2", "lq2", c, spent / k) for p in s["picks"]]
    return rows


def rerank(c: dict, rows: list[dict], build: Path) -> list[dict]:
    """Compile records replace the register estimate; the top few go to the GPU."""
    out = []
    for fam in SHORTLIST:
        got = []
        for r in rows:
            if r["family"] != fam:
                continue
            regs = res_usage(build / r["name"] / "res-usage.txt").get("regs")
            if regs is None:
                continue
            f = family(fam)
            x = r["x"]
            key = key_of(fam, x)
            limit = 255 if x["T"] <= 256 else 128
            need = regs
            if regs >= limit - 5:
                need = max(regs, forms.need_estimate(fam, SHAPES[fam], x))
            a = f.amounts(*key)
            occ = forms.occupancy(x["T"], regs, f.smem(key[0], key[1]))
            a = forms.dataclasses.replace(a, occ=occ, spill=forms.spill_share(need, x["T"]))
            got.append({**r, "regs": regs, "post_us": forms.time_us(a, c[fam])})
        mips = sorted(
            (r for r in got if r["roles"][0].startswith("mip")), key=lambda r: r["post_us"]
        )
        randoms = [r for r in got if r["roles"] == ["random"]][:RANDOM]
        anchors = [r for r in got if r["roles"] == ["anchor"]]
        out += mips[:MEASURE] + randoms + anchors
    return [{**r, "name": re.sub("^lq", "lr", r["name"])} for r in out]


def main() -> None:
    c = json.loads(Path(sys.argv[2]).read_text())
    if sys.argv[1] == "shortlist":
        rows = shortlist(c)
        Path(sys.argv[3]).write_text(json.dumps(rows, indent=0))
    elif sys.argv[1] == "second":
        rows = second(c, json.loads(Path(sys.argv[3]).read_text()))
        Path(sys.argv[4]).write_text(json.dumps(rows, indent=0))
    else:
        rows = rerank(c, json.loads(Path(sys.argv[3]).read_text()), Path(sys.argv[4]))
        Path(sys.argv[5]).write_text(json.dumps(rows, indent=0))
    sys.stdout.write(f"{len(rows)} rows\n")


if __name__ == "__main__":
    main()

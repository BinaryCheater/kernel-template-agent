"""The 24 library operators of WP28 as a breadth set (28 §5, round 10): read from the WP28 worktree,
written as one batch directory of generic kernels, never imported as code.

    python -m kta_perfmodel.breadth build      # perfmodel/batches/r10w/build: SASS, launches
    python -m kta_perfmodel.breadth measured   # after `perfmodel-measure predict r10w` is committed

The SASS with control bits and the launch arguments exist only where WP28 captured them. A case
without control bits stays on the table as `unsupported: <what is missing>` (`unsupported.json`).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from . import upstream

LAB = Path("/Users/dingyi/Projects/agentic-codesign/kta/data/wp28-lab/campaign")
OUT = upstream.ROOT / "perfmodel" / "batches" / "r10w"
GROUP = "breadth library"


def _latest(case: str, kind: str) -> Path | None:
    found = sorted(LAB.glob(f"GB10-suite24-{case}-{kind}-a*"))
    return found[-1] if found else None


def _function(text: str, symbol: str) -> str | None:
    head = f"Function : {symbol}"
    if head not in text:
        return None
    body = text[text.index(head) :]
    cut = body.find("Function : ", len(head))
    return body[: cut if cut > 0 else len(body)]


def _sass(native: Path, symbol: str) -> str | None:
    for p in sorted((native / "capture").glob("module-*.sass")):
        body = _function(p.read_text(errors="replace"), symbol)
        if body:
            return body
    return None


def _launches(native: Path) -> list[dict[str, Any]]:
    ev = [json.loads(x) for x in (native / "capture" / "events.jsonl").read_text().splitlines()]
    by_id: dict[Any, dict[str, Any]] = {}
    for e in ev:  # one launch shows at both the runtime and the driver layer
        if e.get("kind") == "launch" and e.get("window", 1) == 1:
            old = by_id.get(e["correlation_id"])
            if old is None or (e.get("packed_arguments_hex") and not old["packed_arguments_hex"]):
                by_id[e["correlation_id"]] = e
    return list(by_id.values())


def _dim(text: str) -> list[int]:
    return [int(x) for x in text.strip("()").split(",")]


def _bank(hexed: str | None) -> dict[str, int]:
    """The packed arguments as constant-bank words from the first parameter slot."""
    if not hexed:
        return {}
    raw = bytes.fromhex(hexed)
    return {
        str(0x380 + i): int.from_bytes(raw[i : i + 4], "little") for i in range(0, len(raw) - 3, 4)
    }


def build() -> int:
    unsupported: dict[str, str] = {}
    root = OUT / "build" / "testset"
    root.mkdir(parents=True, exist_ok=True)
    for i in range(1, 25):
        case = f"E{i:02d}"
        native, prof = _latest(case, "native"), _latest(case, "profile")
        if prof is None:
            unsupported[case] = "no profile"
            continue
        if native is None or not (native / "capture" / "events.jsonl").exists():
            unsupported[case] = "control bits (no native SASS capture in WP28)"
            continue
        obs = json.loads((prof / "observations.json").read_text())["kernels"]
        launches = _launches(native)
        if len(launches) != len(obs):
            unsupported[case] = f"launches: capture {len(launches)}, profile {len(obs)}"
            continue
        members, res = [], []
        d = root / f"lib_{case}"
        for e, k in zip(launches, obs, strict=True):
            body = _sass(native, e["symbol"])
            if body is None:
                unsupported[case] = f"SASS of {e['symbol'][:40]}"
                break
            block = [x for x in _dim(k["block"])]
            threads = block[0] * block[1] * block[2]
            regs = int(k["launch"]["launch__registers_per_thread"])
            smem = int(float(k["launch"].get("launch__shared_mem_per_block_dynamic") or 0) * 1024)
            members.append(
                {
                    "entry": e["symbol"],
                    "grid": _dim(k["grid"]),
                    "threads": threads,
                    "block": block,
                    "smem": smem,
                    "footprint_bytes": 1 << 20,
                    "write_bytes": 0,
                    "const_bank": _bank(e.get("packed_arguments_hex")),
                    "reads": [],
                    "writes": [],
                }
            )
            res.append((e["symbol"], regs, body))
        else:
            d.mkdir(exist_ok=True)
            seen = {s: b for s, _, b in res}
            (d / "k.sass").write_text("\n".join(seen.values()))
            (d / "kernel.cu").write_text("// library kernel, no source\n")
            (d / "res-usage.txt").write_text(
                "\n".join(f"Function {s}: REG:{r}" for s, r, _ in res) + "\n"
            )
            info = {"family": "library", "set": GROUP, "cache": "warm", "rot": 1, "buffers": {}}
            if len(members) == 1:
                (d / "launch.json").write_text(json.dumps({**members[0], **info}, indent=1))
            else:
                L = {**info, "family": "sequence", "kernels": members, "entry": res[0][0]}
                (d / "launch.json").write_text(json.dumps(L, indent=1))
    (OUT / "unsupported.json").write_text(json.dumps(unsupported, indent=1) + "\n")
    (OUT / "build.json").write_text(json.dumps({"batch": "r10w", "source": str(LAB)}) + "\n")
    print(len(list(root.iterdir())), "cases;", unsupported)
    return 0


def _number(v: Any) -> float | None:
    v = v.get("value", v.get("raw")) if isinstance(v, dict) else v
    try:
        return float(str(v).replace(",", ""))
    except ValueError:
        return None


def measured() -> int:
    """WP28's unprofiled operator medians and its profile's counts; its clock is the batch's."""
    results: dict[str, Any] = {}
    counters: dict[str, Any] = {}
    for d in sorted((OUT / "build" / "testset").iterdir()):
        case = d.name.removeprefix("lib_")
        m = _latest(case, "measure")
        assert m is not None
        t = json.loads((m / "observations.json").read_text())["timing"]
        results[f"{d.name} kernel"] = {"median": t["median_us"], "iqr": 0.0, "unit": "us", "n": 3}
        k = json.loads((_latest(case, "profile") / "observations.json").read_text())["kernels"]  # type: ignore[operator]
        if len(k) == 1:
            raw = {**k[0]["metrics"], **k[0]["launch"]}
            counters[d.name] = {a: _number(v) for a, v in raw.items() if _number(v) is not None}
    doc = {"batch": "r10w", "valid": True, "clock_GHz": None, "results": results}
    doc["counters"] = counters
    (OUT / "measured.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(len(results), "operators")
    return 0


if __name__ == "__main__":
    sys.exit({"build": build, "measured": measured}[sys.argv[1]]())

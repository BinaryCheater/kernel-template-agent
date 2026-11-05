"""Collect one row per measured family kernel: decisions, shape, resources, time, L3 prediction.

Reads `perfmodel/batches/*` read-only. A kernel seen in several batches keeps its newest valid
measurement; `predicted_us` is the L3 prediction committed in that same batch, so it stays a blind
number where the batch was blind.

    python -m tools.kta_bridge.dataset OUT.json
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BATCHES = ROOT / "perfmodel" / "batches"
FAMILIES = {"hgemm", "fa_forward", "gemv_w4a16", "decode_gqa", "rmsnorm", "softmax"}


def batch_order(name: str) -> tuple[int, str]:
    m = re.match(r"r(\d+)(.*)", name)
    return (int(m.group(1)), m.group(2)) if m else (-1, name)


def res_usage(path: Path) -> dict:
    """Registers and shared memory ptxas reported, when its record exists."""
    out: dict = {}
    if not path.exists():
        return out
    text = path.read_text(errors="replace")
    # the last function is the kernel; earlier ones are host-side helpers
    text = text[text.rfind("Function ") :]
    for key, pat in {
        "regs": r"REG:(\d+)",
        "smem": r"SHARED:(\d+)",
        "stack": r"STACK:(\d+)",
    }.items():
        m = re.search(pat, text)
        if m:
            out[key] = int(m.group(1))
    return out


def collect() -> list[dict]:
    rows: dict[str, dict] = {}
    for batch in sorted(
        (p for p in BATCHES.iterdir() if p.is_dir()), key=lambda p: batch_order(p.name)
    ):
        measured = batch / "measured.json"
        if not measured.exists():
            continue
        m = json.loads(measured.read_text())
        if not m.get("valid", True):
            continue
        predicted = {}
        if (batch / "predicted.json").exists():
            predicted = json.loads((batch / "predicted.json").read_text()).get("predicted", {})
        for launch in (batch / "build" / "testset").glob("*/launch.json"):
            name = launch.parent.name
            d = json.loads(launch.read_text())
            if d.get("family") not in FAMILIES:
                continue
            r = m["results"].get(f"{name} kernel")
            if not r:
                continue
            rows[name] = {
                "name": name,
                "batch": batch.name,
                "family": d["family"],
                "shape": d.get("shape", {}),
                "pick": d.get("pick", {}),
                "cache": d.get("cache"),
                "rot": d.get("rot"),
                "set": d.get("set"),
                "grid": d.get("grid"),
                "threads": d.get("threads"),
                "block": d.get("block"),
                "smem": d.get("smem"),
                "footprint_bytes": d.get("footprint_bytes"),
                "write_bytes": d.get("write_bytes"),
                "measured_us": r["median"],
                "iqr_us": r.get("iqr"),
                "predicted_us": predicted.get(f"{name} kernel"),
                "clock_GHz": m.get("clock_GHz"),
                "res": res_usage(launch.parent / "res-usage.txt"),
                "sass": str(launch.parent / "k.sass"),
                "counters": m.get("counters", {}).get(name),
            }
    return sorted(rows.values(), key=lambda r: r["name"])


def main() -> None:
    rows = collect()
    Path(sys.argv[1]).write_text(json.dumps(rows, indent=0))
    sys.stdout.write(f"{len(rows)} rows\n")


if __name__ == "__main__":
    main()

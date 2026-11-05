"""Calibrate the software constants on a dataset and write one frozen constant set per family.

python -m tools.kta_bridge.freeze DATASET.json OUT.json
"""

import json
import sys
from pathlib import Path

from tools.kta_bridge import forms
from tools.kta_bridge.calibrate import fit

HARDWARE = (
    "tensor_w1",
    "tensor_w15",
    "tensor_w2",
    "tensor_w3",
    "copy",
    "store",
    "launch",
    "latency_l2",
    "latency_dram",
)


FAMILIES = ("hgemm", "fa_forward", "decode_gqa", "moe_nvfp4")


def rows_of(dataset: list[dict], family: str) -> list[dict]:
    rows = []
    for r in dataset:
        if r["family"] != family or not r["set"]:
            continue
        p = r["pick"]
        stages = p.get("num_stages", p.get("stages", 0))
        if stages < 1 or not (p.get("block_M") or p.get("BM") or p.get("BN")):
            continue
        if family == "decode_gqa" and not p.get("BN"):
            continue
        if family in ("fa_forward", "decode_gqa") and not r["shape"].get("D"):
            continue
        rows.append({**r, "a": forms.amounts(r)})
    return rows


def main() -> None:
    dataset = json.loads(Path(sys.argv[1]).read_text())
    start = {**forms.DERIVED, **dict.fromkeys(("scalar_w1", "scalar_w15", "scalar_w2"), 1.0)}
    start["scalar_w3"] = 1.0
    out = {f: fit(rows_of(dataset, f), "measured_us", start, HARDWARE) for f in FAMILIES}
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()

"""Worker-side job: query device attributes through the CUDA driver API.

Run as ``{python} -m kta.targets.gb10.devinfo``; writes ``out/devinfo.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from kta.targets.cudadrv import Driver


def main() -> int:
    drv = Driver(0)
    info = {
        "name": drv.name(),
        "driver_version": drv.driver_version(),
        "total_mem_bytes": drv.total_mem(),
        "attributes": drv.attributes(),
    }
    out = Path("out")
    out.mkdir(exist_ok=True)
    (out / "devinfo.json").write_text(json.dumps(info, indent=1, sort_keys=True), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Worker-side job: launch one feature-probe kernel and report whether it ran cleanly.

Run as ``{python} -m kta.targets.gb10.featrun probe.cubin``. The kernel
``kta_probe(unsigned* out)`` runs with one block of 128 threads on a zeroed 4 KiB
buffer; a launch or execution error exits nonzero. ``out/result.json`` holds the first
words of the buffer so the raw effect of the instruction is kept.
"""

from __future__ import annotations

import ctypes
import json
import struct
import sys
from pathlib import Path

from kta.targets.cudadrv import Driver

WORDS = 1024
THREADS = 128


def main(argv: list[str]) -> int:
    cubin = Path(argv[0]).read_bytes()
    drv = Driver(0)
    func = drv.load_function(cubin, "kta_probe")
    buf = drv.alloc(WORDS * 4)
    drv.memset(buf, 0, WORDS * 4)
    drv.launch(func, (1,), (THREADS,), 0, [ctypes.c_uint64(buf.value)])
    drv.synchronize()
    words = struct.unpack(f"<{WORDS}I", drv.to_host(buf, WORDS * 4))
    drv.free(buf)
    out = Path("out")
    out.mkdir(exist_ok=True)
    result = {
        "ran": True,
        "first_words": list(words[:8]),
        "nonzero_words": sum(w != 0 for w in words),
    }
    (out / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

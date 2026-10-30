"""SHA-256 of every raw research-repo file cited by the WP13 analysis outputs.

Lets a reviewer confirm that the read-only inputs are unchanged since the
analysis ran: rerun this script and compare with the committed manifest.

Usage:
    python3 scripts/analysis/f2_inputs_manifest.py [--dir deliveries/WP13]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import f2_common as c

OUTPUTS = ("f2-time-breakdown.json", "f2-model-error.json")


def cited_strings(obj, acc: set[str]) -> None:
    if isinstance(obj, dict):
        for v in obj.values():
            cited_strings(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            cited_strings(v, acc)
    elif isinstance(obj, str) and obj.startswith("kernel-lab/"):
        acc.add(obj.removesuffix(" (mtime)"))


def expand(cited: set[str], root: Path) -> list[Path]:
    files: set[Path] = set()
    for s in cited:
        for part in s.split(" + "):
            part = part.split(" (")[0].strip()
            if "*" in part:
                files.update(p for p in root.glob(part) if p.is_file())
            elif (root / part).is_file():
                files.add(root / part)
    return sorted(files)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=Path("deliveries/WP13"))
    ap.add_argument("--root", type=Path, default=c.RESEARCH_ROOT)
    args = ap.parse_args(argv)
    cited: set[str] = set()
    for name in OUTPUTS:
        cited_strings(c.load_json(args.dir / name), cited)
    files = expand(cited, args.root)
    manifest = {
        "schema": "kta-wp13-inputs-sha256-1",
        "root": str(args.root),
        "files": {c.rel(p, args.root): sha256(p) for p in files},
    }
    c.write_json(args.dir / "f2-inputs-sha256.json", manifest)
    sys.stdout.write(f"hashed {len(files)} input files\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

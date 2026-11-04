"""Verify retained CUTLASS archive and every pinned Git blob without fetching."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    archive, root = Path(manifest["archive"]), Path(manifest["tree"])
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["archive_sha256"]
    tree = {}
    for line in (STUDY / "inputs/git-tree.txt").read_text().splitlines():
        meta, path = line.split("\t")
        _, kind, blob = meta.split()
        assert kind == "blob"
        tree[path] = blob
    assert set(tree) == {entry["path"] for entry in manifest["files"]}
    for entry in manifest["files"]:
        data = (root / entry["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"]
        blob = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        assert blob == tree[entry["path"]] == entry["git_blob"]
    result = {
        "outcome": "PASS",
        "blob_count": len(tree),
        "archive": str(archive),
        "tree": str(root),
        "missing": [],
        "mismatches": [],
    }
    (STUDY / "inputs/retention-check.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(f"PASS: retained archive and all {len(tree)} CUTLASS blobs.\n")


if __name__ == "__main__":
    main()

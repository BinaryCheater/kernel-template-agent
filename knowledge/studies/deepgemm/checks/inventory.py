"""Reconcile retained superproject files and enumerate review entry points."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    root = Path(manifest["tree"])
    expected = {item["path"]: item["sha256"] for item in manifest["files"]}
    actual = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }
    assert actual == expected
    tree_lines = (STUDY / "inputs/git-tree.txt").read_text().splitlines()
    git_files = {line.split("\t", 1)[1] for line in tree_lines if line.startswith("100")}
    assert set(actual) == git_files
    for line in tree_lines:
        if line.startswith("100"):
            header, name = line.split("\t", 1)
            payload = (root / name).read_bytes()
            blob = b"blob " + str(len(payload)).encode() + b"\0" + payload
            assert hashlib.sha1(blob).hexdigest() == header.split()[2]
    gitlinks = [line.split("\t", 1)[1] for line in tree_lines if line.startswith("160000")]
    registrations = []
    devices = []
    for name in sorted(actual):
        text = (root / name).read_text()
        if name.startswith("csrc/apis/"):
            for match in re.finditer(r'\b(?:m|submodule)\.def\(\s*"([^"]+)"', text):
                registrations.append(
                    {
                        "path": name,
                        "line": text[: match.start()].count("\n") + 1,
                        "name": match.group(1),
                    }
                )
        if (
            (name.startswith("deep_gemm/include/") and name.endswith(".cuh"))
            or (name.startswith("deep_gemm/legacy/") and name.endswith(".py"))
            or name == "csrc/indexing/main.cu"
            or name.startswith("third-party/tilelang_ops/")
        ):
            if (
                "CUTLASS_GLOBAL" in text
                or "__global__" in text
                or "@triton.jit" in text
                or "@T.prim_func" in text
            ):
                devices.append(name)
    retained = json.loads((STUDY / "inputs/submodule-sources.json").read_text())
    assert {item["superproject_path"] for item in retained} == set(gitlinks)
    for item in retained:
        assert any(
            item["revision"] in line and line.endswith(item["superproject_path"])
            for line in tree_lines
        )
        archive = Path(item["archive"])
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == item["archive_sha256"]
        for used in item["used_files"]:
            payload = (Path(item["tree"]) / used["path"]).read_bytes()
            assert hashlib.sha256(payload).hexdigest() == used["sha256"]
    result = {
        "outcome": "PASS",
        "superproject_files": len(actual),
        "retained_direct_submodules": gitlinks,
        "missing_direct_submodule_bodies": [],
        "api_registrations": registrations,
        "device_source_files": devices,
        "boundary": "File inventory and entry discovery; semantic group coverage remains open.",
    }
    (STUDY / "checks/inventory-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        f"PASS: {len(actual)} files; {len(registrations)} API registrations; "
        f"{len(devices)} device source files."
    )


if __name__ == "__main__":
    main()
    sys.stdout.write("\n")

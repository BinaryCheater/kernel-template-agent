"Reconcile the pinned source's exports, builders and device entries without importing it."

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def group_for(path: str) -> list[str]:
    "Assign source ownership; shared includes intentionally have multiple consumers."
    if path.endswith("kernel/humming.cuh"):
        return ["G01", "G02", "G04"]
    if path.endswith("kernel/humming_ws.cuh"):
        return ["G01", "G02", "G04"]
    if path.endswith("kernel/humming_umma.cuh") or "/mma/umma" in path:
        return ["G03"]
    if "/mma/wmma" in path:
        return ["G01"]
    if "/mma/wgmma" in path:
        return ["G02"]
    if "/mma/mxmma" in path:
        return ["G04"]
    if path.endswith("scheduler.cuh"):
        return ["G05"]
    if "/epilogue/" in path or "/arith/epilogue" in path:
        return ["G06"]
    if "/datatype/" in path or "/arith/" in path:
        return ["G07", "G10"]
    if path.endswith("kernel/process.cuh"):
        return ["G08"]
    if path.endswith("kernel/pack_weight.cuh"):
        return ["G09"]
    if path.endswith(("kernel/quant_weight.cuh", "kernel/dequant_weight.cuh")):
        return ["G10"]
    if "/kernel/process_input" in path:
        return ["G11"]
    if path.endswith("kernel/process_mxfp4.cuh"):
        return ["G12"]
    if path.endswith("kernel/tops_bench.cuh"):
        return ["G13"]
    if "/memory/" in path or "/utils/" in path or path.endswith("mma/all.cuh"):
        return ["shared_infrastructure"]
    raise ValueError(f"Unassigned device source: {path}")


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    source = Path(manifest["humming"]["root"])
    frozen = {row["path"]: row for row in manifest["humming"]["files"]}
    actual = {str(p.relative_to(source)) for p in source.rglob("*") if p.is_file()}
    assert actual == set(frozen), "Pinned source file set changed"
    for path, row in frozen.items():
        assert hashlib.sha256((source / path).read_bytes()).hexdigest() == row["sha256"], path
    tree = ast.parse((source / "humming/ops/__init__.py").read_text())
    exports = next(
        ast.literal_eval(node.value)
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets)
    )
    builders = []
    for path in sorted((source / "humming/kernel").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ClassDef) and any(
                isinstance(base, ast.Name) and base.id in ("KernelRuntime", "ProcessInputKernel")
                for base in node.bases
            ):
                builders.append(
                    {"path": str(path.relative_to(source)), "line": node.lineno, "name": node.name}
                )
    headers, entries, variants = [], [], []
    for path in sorted((source / "humming/include").rglob("*.cuh")):
        relative = str(path.relative_to(source))
        groups = group_for(relative)
        headers.append({"path": relative, "groups": groups})
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if "__global__" in line:
                match = re.search(r"\bvoid\s+(\w+)\s*\(", line)
                assert match, f"Unrecognized device entry: {relative}:{number}"
                entries.append(
                    {"path": relative, "line": number, "name": match[1], "groups": groups}
                )
            if re.search(r"\bif\s+constexpr\s*\(|\bstatic_assert\s*\(|^\s*#\s*(if|elif)\b", line):
                variants.append(
                    {
                        "path": relative,
                        "line": number,
                        "source": line.strip(),
                        "groups": groups,
                        "disposition": "requires group trace; not coverage credit",
                    }
                )
    result = {
        "exports": exports,
        "builders": builders,
        "device_entries": entries,
        "headers": headers,
        "variant_anchors": variants,
        "method": (
            "AST exports/builders; every CUDA header; all __global__ declarati"
            "ons; line anchors for compile-time branches/assertions (multiline"
            " text is partial)."
        ),
    }
    (STUDY / "checks/inventory.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        f"Source integrity PASS: {len(actual)} files. Inventory: {len(exports)} exports, "
        f"{len(builders)} builders, {len(entries)} device entries, {len(headers)} headers, "
        f"{len(variants)} branch/assertion anchors. Semantic trace status is separate."
    )


if __name__ == "__main__":
    main()
    sys.stdout.write("\n")

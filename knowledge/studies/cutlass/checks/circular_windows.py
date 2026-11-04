"""Recover SM103's eight descriptor pairs and replay cross-buffer K windows."""

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def coordinate(node: ast.Tuple) -> tuple[int, int]:
    offset = ast.literal_eval(node.elts[2])
    buffer = node.elts[3]
    assert isinstance(buffer, ast.Attribute) and isinstance(buffer.value, ast.Name)
    return int(buffer.value.id[-1]), offset


def extract_source_pairs(manifest: dict, path: str) -> tuple[list, dict]:
    data = (Path(manifest["tree"]) / path).read_bytes()
    entry = next(x for x in manifest["files"] if x["path"] == path)
    assert hashlib.sha256(data).hexdigest() == entry["sha256"]
    tree = ast.parse(data)
    kernel = next(
        x for x in ast.walk(tree) if isinstance(x, ast.FunctionDef) and x.name == "kernel"
    )
    events = sorted(
        (x for x in ast.walk(kernel) if isinstance(x, (ast.Assign, ast.Call))),
        key=lambda n: n.lineno,
    )
    pairs, release = [], {}
    current = following = None
    for event in events:
        if isinstance(event, ast.Assign) and len(event.targets) == 1:
            name = event.targets[0]
            if isinstance(name, ast.Name) and name.id == "k_block_coord_cur":
                current = coordinate(event.value)
            elif isinstance(name, ast.Name) and name.id == "k_block_coord_next":
                following = coordinate(event.value)
        elif isinstance(event, ast.Call) and isinstance(event.func, ast.Attribute):
            if event.func.attr == "make_desc_and_call_mma":
                assert current is not None and following is not None
                pairs.append((current, following))
            elif event.func.attr == "release" and isinstance(event.func.value, ast.Name):
                name = event.func.value.id
                if name.startswith("ab_full"):
                    release[int(name[-1])] = len(pairs) - 1
    return pairs, release


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    path = (
        "examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/sm"
        "103_dense_blockscaled_gemm_persistent.py"
    )
    pairs, release = extract_source_pairs(manifest, path)
    grouped_path = (
        "examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_grouped"
        "_gemm/sm103_grouped_blockscaled_gemm.py"
    )
    grouped_pairs, grouped_release = extract_source_pairs(manifest, grouped_path)
    assert grouped_pairs == pairs and grouped_release == release
    assert len(pairs) == 8 and release == {0: 2, 1: 5, 2: 7}
    for i, (cur, nxt) in enumerate(pairs):
        start = cur[0] * 256 + cur[1] * 32
        assert start == i * 96
        actual = []
        for k in range(96):
            position = cur[1] * 32 + k
            value = cur[0] * 256 + position if position < 256 else nxt[0] * 256 + position - 256
            actual.append(value)
        assert actual == list(range(i * 96, (i + 1) * 96))
    for buf, last_use in release.items():
        users = [
            i
            for i in range(8)
            if set(range(i * 96, (i + 1) * 96)).intersection(range(buf * 256, (buf + 1) * 256))
        ]
        assert max(users) == last_use
    scale_groups = 0
    for vector in (16, 32):
        buffers = 4 if vector == 16 else 2
        per_buffer, sf_stride = 8 // buffers, 6 if vector == 16 else 3
        for mma, elem in itertools.product(range(8), range(96)):
            stage, mma_in_stage = divmod(mma, per_buffer)
            group = stage * 12 + mma_in_stage * sf_stride + elem // vector
            assert group == (mma * 96 + elem) // vector
            scale_groups += 1
    generalized = 0
    for chunk, window, iteration in itertools.product(range(1, 33), range(1, 33), range(20)):
        if window > chunk:
            continue
        start = iteration * window
        buf, offset = divmod(start, chunk)
        mapped = [
            (buf + ((offset + k) // chunk)) * chunk + ((offset + k) % chunk) for k in range(window)
        ]
        assert mapped == list(range(start, start + window))
        assert all(x // chunk in (buf, buf + 1) for x in mapped)
        generalized += 1
    result = {
        "outcome": "PASS",
        "source_descriptor_pairs": pairs,
        "grouped_source_pair_match": True,
        "last_use_release": release,
        "scale_group_coordinates": scale_groups,
        "bounded_window_domains": generalized,
        "witness": {"instruction_k": 96, "buffer_k": 256, "macro_k": 768, "crossing_mmas": [2, 5]},
        "boundary": (
            "Logical descriptor/window and scale ownership; hardware encoding/order open."
        ),
    }
    (STUDY / "checks/circular-windows-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        "PASS: eight SM103 descriptor windows, release frontiers and independent SF chunks.\n"
    )


if __name__ == "__main__":
    main()

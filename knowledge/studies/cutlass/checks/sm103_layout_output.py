"Check SM103 scale address composition with retained pure-Python CuTe algebra."

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    root = Path(manifest["tree"])
    for entry in manifest["files"]:
        if entry["path"].startswith("python/pycute/"):
            assert (
                hashlib.sha256((root / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
            )
    sys.dont_write_bytecode = True
    package = root / "python/pycute"
    spec = importlib.util.spec_from_file_location(
        "study_pycute", package / "__init__.py", submodule_search_locations=[str(package)]
    )
    assert spec is not None and spec.loader is not None
    cute = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = cute
    spec.loader.exec_module(cute)
    atom_positions = staged_positions = 0
    for vector in (16, 32):
        atom = cute.Layout(((8, 4, 4), (vector, 4)), ((16, 128, 4), (0, 1)))
        ordinary = cute.Layout(((32, 4), (vector, 4)), ((16, 4), (0, 1)))
        coalesced = cute.coalesce(atom[0])
        assert coalesced.shape == (32, 4) and coalesced.stride == (16, 4)
        for row, k in itertools.product(range(128), range(vector * 4)):
            a = atom(((row % 8, row // 8 % 4, row // 32), (k % vector, k // vector)))
            b = ordinary(((row % 32, row // 32), (k % vector, k // vector)))
            assert a == b
            atom_positions += 1
        # Source one-SF-buffer contains twelve groups per row, for either vector size.
        for width in (128, 256):
            if width == 128:
                layout = cute.tiled_product(atom, cute.Layout((1, 3)))
            else:
                atom256 = cute.Layout(((32, 4, 2), (vector, 4)), ((16, 4, 1536), (0, 1)))
                layout = cute.tiled_product(atom256, cute.Layout((1, 3)))
            assert cute.cosize(layout) == width * 12
            tags = {}
            for row, group in itertools.product(range(width), range(12)):
                if width == 128:
                    mn_coord = (row % 8, row // 8 % 4, row // 32)
                else:
                    mn_coord = (row % 32, row // 32 % 4, row // 128)
                index = layout(((mn_coord, (0, group % 4)), 0, group // 4))
                expected = (row % 32) * 16 + (row // 32 % 4) * 4 + group % 4
                expected += (group // 4) * 512 + (row // 128) * 1536
                assert index == expected and index not in tags
                tags[index] = (row, group)
            assert set(tags) == set(range(width * 12))
            for stages in (1, 2, 5):
                occupied = set()
                for stage, index in itertools.product(range(stages), tags):
                    physical = stage * width * 12 + index
                    assert physical not in occupied
                    occupied.add(physical)
                    staged_positions += 1
    # The direct epilogue guards a vector by its first coordinate. Check its
    # semantic premise: the contiguous valid extent is a multiple of vector size.
    packets = 0
    for tile, valid, vector in itertools.product((128, 256), range(0, 257, 8), (1, 2, 4, 8)):
        if valid > tile or valid % vector:
            continue
        actual = [
            x
            for start in range(0, tile, vector)
            if start < valid
            for x in range(start, start + vector)
        ]
        assert actual == list(range(valid))
        packets += 1
    # A partial vector invalidates a first-coordinate-only guard.
    assert [x for start in range(0, 16, 8) if start < 10 for x in range(start, start + 8)] != list(
        range(10)
    )
    # The caller pre-advances its tile count before entering the TMA delegate.
    ring_cases = 0
    for subtiles, stages in itertools.product((4, 8), (1, 2, 3, 5)):
        observed = [
            (executed * subtiles + subtile) % stages
            for executed in range(1, 6)
            for subtile in range(subtiles)
        ]
        expected = [(subtiles + i) % stages for i in range(5 * subtiles)]
        assert observed == expected
        ring_cases += 1
    store_cases = 0
    for stages, tiles in itertools.product(range(1, 9), range(33)):
        slots, unread, captured = {}, [], {}
        for tile in range(tiles):
            slot = tile % stages
            assert all(pending_slot != slot for _, pending_slot in unread)
            slots[slot] = tile
            unread.append((tile, slot))
            while len(unread) > stages - 1:
                group, pending_slot = unread.pop(0)
                captured[group] = slots[pending_slot]
        for group, slot in unread:
            captured[group] = slots[slot]
        # Destination writes may finish later and in another order; source values
        # are already captured before their shared slots are overwritten.
        output = {group: captured[group] for group in reversed(range(tiles))}
        assert output == {tile: tile for tile in range(tiles)}
        store_cases += 1
    unread_slots = {}
    unsafe_witness = None
    for tile in range(2):
        slot = tile % 1
        if slot in unread_slots:
            unsafe_witness = {
                "slot": slot,
                "unread_generation": unread_slots[slot],
                "overwrite_generation": tile,
                "wait_limit": 1,
            }
            break
        unread_slots[slot] = tile
        # The weakened wait allows one unread group, so generation0 is not retired.
    assert unsafe_witness == {
        "slot": 0,
        "unread_generation": 0,
        "overwrite_generation": 1,
        "wait_limit": 1,
    }
    result = {
        "outcome": "PASS",
        "atom_address_relations": atom_positions,
        "staged_scale_positions": staged_positions,
        "whole_vector_tail_domains": packets,
        "tma_ring_domains": ring_cases,
        "source_read_store_domains": store_cases,
        "weaker_wait_counterexample": unsafe_witness,
        "boundary": (
            "Retained PyCuTe algebra and source address equations; no DSL lowering or GPU run."
        ),
    }
    (STUDY / "checks/sm103-layout-output-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    sys.stdout.write(
        "PASS: SM103 atom/host scale equivalence, staged SF extents and output packet premise.\n"
    )


if __name__ == "__main__":
    main()

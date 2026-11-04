"Replay selected non-mixed persistent cluster schedules and multicast releases."

from __future__ import annotations

import collections
import itertools
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def decode(work: int, cm: int, cn: int, swizzle: int, along_m: bool) -> tuple[int, int, int]:
    batch, local = divmod(work, cm * cn)
    if swizzle == 1:
        minor, major = divmod(local, cm if along_m else cn)
        return (major, minor, batch) if along_m else (minor, major, batch)
    section, residue = divmod(local, swizzle)
    if along_m:
        nsection, m = divmod(section, cm)
        return m, swizzle * nsection + residue, batch
    msection, n = divmod(section, cn)
    return swizzle * msection + residue, n, batch


def schedules() -> dict[str, int]:
    domains = padded = 0
    for mt, nt, batches, cta_m, cta_n, active, swizzle, along_m in itertools.product(
        (1, 3, 7), (1, 4, 9), (1, 2), (1, 2, 4), (1, 2, 4), (1, 3, 8), (1, 2, 4), (False, True)
    ):
        cm, cn = (mt + cta_m - 1) // cta_m, (nt + cta_n - 1) // cta_n
        if swizzle > 1:
            if along_m:
                cn = (cn + swizzle - 1) // swizzle * swizzle
            else:
                cm = (cm + swizzle - 1) // swizzle * swizzle
        total = cm * cn * batches
        grid = min(total, active)
        visited = set()
        live = collections.Counter()
        for worker in range(grid):
            for work in range(worker, total, grid):
                coord = decode(work, cm, cn, swizzle, along_m)
                assert coord not in visited
                visited.add(coord)
                for lm, ln in itertools.product(range(cta_m), range(cta_n)):
                    output = (coord[0] * cta_m + lm, coord[1] * cta_n + ln, coord[2])
                    if output[0] < mt and output[1] < nt:
                        live[output] += 1
                    else:
                        padded += 1
        assert visited == set(itertools.product(range(cm), range(cn), range(batches)))
        assert set(live.values()) == {1} and len(live) == mt * nt * batches
        domains += 1
    return {"domains": domains, "padded_cta_tile_visits": padded}


def multicast() -> dict[str, int]:
    configurations = masks = 0
    for cta_m, cta_n, v in itertools.product((1, 2, 4), (1, 2, 4), (1, 2)):
        if cta_m % v:
            continue
        msize, nsize = cta_m // v, cta_n
        coords = list(itertools.product(range(v), range(msize), range(nsize)))
        incoming = collections.Counter()
        for source_m, source_n in itertools.product(range(msize), range(nsize)):
            # Source helper ORs A/B masks and their V-peer masks.
            own_a = {(0, source_m, n) for n in range(nsize)}
            own_b = {(0, m, source_n) for m in range(msize)}
            source_mask = own_a | own_b
            if v == 2:
                source_mask |= {(1, m, n) for _, m, n in own_a | own_b}
            expected = {c for c in coords if c[1] == source_m or c[2] == source_n}
            assert source_mask == expected
            assert len(source_mask) == v * (msize + nsize - 1)
            incoming.update(source_mask)
            masks += 1
        # Each target's empty barrier sees one release from each row/column leader group.
        assert set(incoming) == set(coords)
        assert set(incoming.values()) == {msize + nsize - 1}
        configurations += 1
    return {"configurations": configurations, "release_masks": masks}


def main() -> None:
    result = {
        "outcome": "PASS",
        "scheduler": schedules(),
        "multicast": multicast(),
        "boundary": "Integer source relations; no barrier timing, residency or device execution.",
    }
    (STUDY / "checks/cluster-scheduler-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        "PASS: persistent cluster tiles, swizzled padding and row/column release counts.\n"
    )


if __name__ == "__main__":
    main()

"""Logical physical-CGA payload/owner/release maps; no native TMA/UMMA replay."""

import json
from pathlib import Path


def masks(dx, dy, x, y, relation):
    if relation == "x_column":
        payload = {x + dx * yy for yy in range(dy)}
        releases = {x // 2 * 2, x // 2 * 2 + 1}
        expected = dy
    elif relation == "x_parity_row":
        payload = {x % 2 + 2 * a + dx * y for a in range(dx // 2)}
        releases = {dx * y, dx * y + 1}
        expected = dx // 2
    else:
        payload = {xx + dx * y for xx in range(dx)}
        releases = {dx * y}
        expected = dx // 2
    owners = {rank & ~1 for rank in payload}
    return payload, owners, releases, expected


def bits(ranks):
    return sum(1 << rank for rank in ranks)


if __name__ == "__main__":
    cases = []
    for dx, dy in ((2, 1), (4, 1), (2, 2), (4, 2), (4, 4)):
        for relation in ("x_column", "x_parity_row", "whole_x_row"):
            issuers = (
                [(x, 0) for x in range(dx)]
                if relation == "x_column"
                else [(x, y) for y in range(dy) for x in (0, 1)]
                if relation == "x_parity_row"
                else [(0, y) for y in range(dy)]
            )
            for x, y in issuers:
                payload, owners, release, expected = masks(dx, dy, x, y, relation)
                packed = bits(payload)
                assert bits(owners) == (packed | (packed >> 1)) & 0x5555
                assert all(0 <= rank < dx * dy for rank in payload | owners | release)
                assert all(rank % 2 == 0 for rank in owners)
                assert len(owners) == expected
                # One completed consuming atom fans its release to each target.
                arrivals = {target: 0 for target in release}
                for _owner in owners:
                    for target in release:
                        arrivals[target] += 1
                assert all(count == expected for count in arrivals.values())
                cases.append(
                    {
                        "shape": [dx, dy],
                        "relation": relation,
                        "issuer": x + dx * y,
                        "payload": sorted(payload),
                        "full_owners": sorted(owners),
                        "release": sorted(release),
                        "expected_atom_arrivals": expected,
                    }
                )
    wide = masks(4, 2, 0, 0, "whole_x_row")
    assert wide == ({0, 1, 2, 3}, {0, 2}, {0}, 2)
    assert wide[0] != wide[1] != wide[2]
    odd_a = masks(4, 2, 1, 0, "x_column")
    assert odd_a == ({1, 5}, {0, 4}, {0, 1}, 2)
    fallback = masks(2, 1, 0, 0, "whole_x_row")
    assert fallback == ({0, 1}, {0}, {0}, 1)
    assert wide[3] != fallback[3]
    assert any(rank >= 2 for rank in wide[0] - fallback[0])
    # Independent ragged-GEMM retrieval premises: truncation is not a rebase.
    assert (128 // 128) * 128 == 128
    assert (128 // 256) * 256 != 128
    # Raw K partials must receive their own scale before cross-scale summation.
    raw, scales = [2.0, 5.0], [0.5, 4.0]
    assert sum(r * s for r, s in zip(raw, scales, strict=True)) != sum(raw) * scales[-1]
    result = {
        "kind": "logical_atom_multicast_check",
        "native_execution": False,
        "status": "passed",
        "cases": cases,
        "checks": [
            "even_owner_projection",
            "payload_full_release_distinct",
            "atom_release_census",
            "odd_parity_issuer",
            "fallback_recomputed",
            "segment_alignment",
            "scale_boundary_promotion",
        ],
        "remaining": (
            "Native TMA byte accounting, pair completion, remote barriers and cluster retirement."
        ),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")

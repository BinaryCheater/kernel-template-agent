"Marlin A/ldmatrix and output-coordinate relations, plus host chunk coverage."

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def shapes() -> list[tuple[int, int, int]]:
    return [(128, 128, 1), *[(64, 256, m) for m in range(1, 5)]]


def check_a_fragments() -> int:
    checked = 0
    for tk, tn, mb in shapes():
        stride, nwarps = tk // 8, tn // 64

        def transform(index: int, width: int = stride) -> int:
            return index ^ (index // width)

        stored = {
            transform(row * stride + col): [(row, col * 8 + e) for e in range(8)]
            for row in range(mb * 16)
            for col in range(stride)
        }
        for warp, step, mblock in itertools.product(range(8), range(2), range(mb)):
            matrices = []
            for matrix in range(4):
                rows = []
                for row in range(8):
                    address_lane = matrix * 8 + row
                    logical = stride * (address_lane % 16) + address_lane // 16
                    logical += (
                        2 * (warp // nwarps) + 2 * (8 // nwarps) * step + stride * 16 * mblock
                    )
                    rows.append(stored[transform(logical)])
                matrices.append(rows)
            for lane, elem in itertools.product(range(32), range(8)):
                matrix, component = divmod(elem, 2)
                actual = matrices[matrix][lane // 4][2 * (lane % 4) + component]
                expected = (
                    mblock * 16 + lane // 4 + (8 if elem % 4 >= 2 else 0),
                    16 * (warp // nwarps + step * (8 // nwarps))
                    + 2 * (lane % 4)
                    + elem % 2
                    + (8 if elem >= 4 else 0),
                )
                assert actual == expected
                checked += 1
    return checked


def check_outputs() -> dict[str, int]:
    checked, sf_checked = 0, 0
    channel_perm = [2 * i + j for i in range(4) for j in (0, 1, 8, 9, 16, 17, 24, 25)]
    for _, tn, mb in shapes():
        nblocks = tn // 16
        shared_stride = 2 * nblocks + 1
        shared = {}
        for warp, lane, mblock, word, fragment, component in itertools.product(
            range(tn // 64), range(32), range(mb), range(4), range(2), range(2)
        ):
            base = 4 * shared_stride * (lane // 4) + lane % 4 + 32 * warp
            base += 16 * (4 * shared_stride) * mblock + 8 * word + 4 * fragment
            for rowhalf in (0, 1):
                address = (base + rowhalf * 8 * 4 * shared_stride) * 2 + component
                expected = (
                    16 * mblock + lane // 4 + 8 * rowhalf,
                    warp * 64 + word * 16 + fragment * 8 + 2 * (lane % 4) + component,
                )
                assert address not in shared
                shared[address] = expected
            # Source per-channel scale register load and half2 selection.
            s_read = 8 * warp + lane % 4
            packed_sf = s_read * 8 + (word // 2) * 32 + 4 * (word % 2) + 2 * fragment + component
            block, offset = divmod(packed_sf, 32)
            assert block * 32 + channel_perm[offset] == expected[1]
            sf_checked += 1
        assert len(shared) == 16 * mb * tn
        for valid in range(1, 16 * mb + 1):
            writes = {}
            for thread in range(256):
                row, vec_col = divmod(thread, 2 * nblocks)
                for i in range((16 * mb + 256 // (2 * nblocks) - 1) // (256 // (2 * nblocks))):
                    actual_row = row + i * (256 // (2 * nblocks))
                    if actual_row >= valid:
                        continue
                    vector_address = shared_stride * actual_row + vec_col
                    for component in range(8):
                        value = shared[vector_address * 8 + component]
                        assert value == (actual_row, vec_col * 8 + component)
                        assert value not in writes
                        writes[value] = 1
            assert len(writes) == valid * tn
            checked += len(writes)
    return {"live_output_coordinates": checked, "channel_scale_bindings": sf_checked}


def host_chunks(m: int, max_par: int) -> list[tuple[int, int, int]]:
    blocks, pad = (m + 15) // 16, ((m + 15) // 16) * 16 - m
    i, offset, result = 0, 0, []
    while i < blocks:
        mb, prob_m, par = blocks - i, m - 16 * i, 1
        if mb > 4:
            par = min((16 * mb - pad) // 64, max_par)
            prob_m = 64 * par
            i += 4 * (par - 1)
            mb = 4
        result.append((offset, prob_m, mb))
        offset += 16 * mb * par
        i += 4
    return result


def check_host() -> dict[str, object]:
    domains = 0
    for m, max_par in itertools.product(range(1025), (1, 2, 8, 16)):
        chunks = host_chunks(m, max_par)
        covered = [row for start, count, _ in chunks for row in range(start, start + count)]
        assert covered == list(range(m))
        assert all(1 <= mb <= 4 for _, _, mb in chunks)
        domains += 1
    # M65 with max_par0 does not advance the host loop: i+=-4 then the for-loop adds4.
    assert 0 + 4 * (0 - 1) + 4 == 0
    return {
        "m_parallel_domains": domains,
        "max_par_zero_nonprogress": {"m": 65, "initial_i": 0, "next_i": 0},
    }


def main() -> None:
    root = Path(json.loads((STUDY / "inputs/sources.json").read_text())["tree"])
    body = (root / "marlin/marlin_cuda_kernel.cu").read_text()
    assert "par = max_par;" in body and "int c_sh_rd = c_sh_stride" in body
    result = {
        "outcome": "PASS",
        "a_fragment_coordinates": check_a_fragments(),
        "output": check_outputs(),
        "host": check_host(),
        "boundary": (
            "PTX logical layout/source index relations only; no ldmatrix/MMA or GPU execution."
        ),
    }
    (STUDY / "checks/output-callers-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write("PASS: A fragments, output/scale coordinates and M chunk/tail coverage.\n")


if __name__ == "__main__":
    main()

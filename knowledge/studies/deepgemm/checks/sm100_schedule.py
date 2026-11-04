"Check SM100 scale ownership and adjacent accumulator-stage aliasing equations."

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def scale_rows(block_k: int, umma_k: int, gran: int, blocks: int) -> list[list[int]]:
    loads_every = 1 if gran == 32 else 4
    result = []
    for block in range(blocks):
        loaded_row = block // loads_every * (block_k // 128)
        for offset in range(0, block_k, umma_k):
            subblock = offset // 128
            sf_id = offset % 128 // 32 if gran == 32 else block % loads_every
            # TMEM selects a byte within each four-byte word.
            first = (loaded_row + subblock) * 4 + sf_id
            result.append([first + i if gran == 32 else first for i in range(umma_k // 32)])
    return result


def verify_alias(width: int, sf_cols: int, iterations: int) -> int:
    overlap = max(2 * width + sf_cols, 512) - 512
    assert 0 <= overlap <= width
    starts = [0, width - overlap]
    sets = [set(range(start, start + width)) for start in starts]
    assert len(sets[0] & sets[1]) == overlap
    sf_start = 2 * width - overlap
    assert max(sets[0] | sets[1]) < sf_start
    assert sf_start + sf_cols <= 512
    hazards_rejected = 0
    for iteration in range(iterations):
        stage = iteration % 2
        reverse = stage == 0 and overlap > 0
        read_order = list(range(width - 1, -1, -1)) if reverse else list(range(width))
        first_read = {starts[stage] + i for i in read_order[:overlap]}
        if overlap:
            assert first_read == sets[0] & sets[1]
            # Before the last required read, releasing overlap permits destruction.
            assert set(first_read) - {starts[stage] + i for i in read_order[: overlap - 1]}
            hazards_rejected += 1
        # The other stage can overwrite only its overlap after that subset was read.
        # The same stage's next generation still waits for every load to finish.
        assert {starts[stage] + i for i in read_order} == sets[stage]
    return hazards_rejected


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    root = Path(manifest["tree"])
    body = (root / "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_gemm_1d1d.cuh").read_text()
    epilogue = (root / "deep_gemm/include/deep_gemm/epilogue/sm100_store_cd.cuh").read_text()
    for marker in (
        "kIsMXF4 ? 64 : 32",
        "kGranKA == 32 ? 1 : 4",
        "kGranKB == 32 ? 1 : 4",
        "offset / kNumUTCCPAlignedElems",
        "kNumAccumTmemCols + kNumSFTmemCols, 512",
        "smem.tmem_overlap_barriers[preceding_stage_idx].wait(preceding_phase_idx)",
        "umma_k_idx > 0 or k_block_idx > 0",
    ):
        assert marker in body
    assert "accum_stage_idx == 0" in body
    assert "i + 1 == kNumOverlapLoads" in epilogue
    selectors = []
    # Mixed FP8/FP4 uses unpacked shared operands and K32 UMMA; pure FP4 uses packed K64.
    for gran_a, gran_b in itertools.product((32, 128), repeat=2):
        for gran in (gran_a, gran_b):
            actual = scale_rows(128, 32, gran, 9)
            expected = [
                [(block * 128 + offset) // gran]
                for block in range(9)
                for offset in range(0, 128, 32)
            ]
            assert actual == expected
        selectors.append(
            {
                "kind": "mxf8f6f4",
                "gran_a": gran_a,
                "gran_b": gran_b,
                "blocks": 9,
                "selector_rows_a": scale_rows(128, 32, gran_a, 9),
            }
        )
    fp4 = scale_rows(256, 64, 32, 5)
    expected_fp4 = [
        [(block * 256 + offset + half * 32) // 32 for half in range(2)]
        for block in range(5)
        for offset in range(0, 256, 64)
    ]
    assert fp4 == expected_fp4
    assert fp4[1] == [2, 3] and fp4[2] == [4, 5]
    alias_domains, hazards = 0, 0
    for width, sf_cols in itertools.product((64, 128, 192, 248, 256), (8, 12, 16)):
        hazards += verify_alias(width, sf_cols, 6)
        alias_domains += 1
    # Concrete non-swap N256: SF_A/B each4 columns -> overlap8, starts0/248, SF504..511.
    assert max(2 * 256 + 8, 512) - 512 == 8
    # One signal for a partially read stage cannot stand for the full-stage release.
    assert 8 < 256
    result = {
        "outcome": "PASS",
        "mixed_scale_selectors": selectors,
        "pure_fp4_k64_selectors": fp4,
        "alias_domains": alias_domains,
        "early_release_hazards_rejected": hazards,
        "tmem_witness": {
            "N": 256,
            "sf_columns": 8,
            "overlap_columns": 8,
            "stage_starts": [0, 248],
            "sf_start": 504,
        },
        "open": [
            "MMA and UTCCP numerical/hardware layout execution",
            "two-CTA async proxy and arrival-count proof",
            "complete epilogue fragment map",
        ],
        "boundary": (
            "Source-index and interval equations only; no CUDA execution or runtime race proof."
        ),
    }
    (STUDY / "checks/sm100-schedule-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        f"PASS: 4 mixed-granularity domains, pure FP4 K64, {alias_domains} alias domains.\n"
    )


if __name__ == "__main__":
    main()

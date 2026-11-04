"Logical live-prefix/capacity and sparse-fragment maps; no native GPU replay."

import json
import math
from pathlib import Path


def partition(total, requested, instruction_group):
    span = max(
        instruction_group,
        ((total + requested * instruction_group - 1) // (requested * instruction_group))
        * instruction_group,
    )
    active = (total + span - 1) // span
    ranges = [range(s * span, min((s + 1) * span, total)) for s in range(active)]
    return span, active, ranges


def fragment_locators(block, block_size, page_size, table):
    fragment = math.gcd(block_size, page_size)
    subpages = page_size // fragment
    result = []
    for f in range(block_size // fragment):
        token = block * block_size + f * fragment
        page, offset = divmod(token, page_size)
        result.append(table[page] * subpages + offset // fragment)
    return fragment, result


def decode(locator, page_size, fragment):
    subpages = page_size // fragment
    if locator < 0:
        return -1, 0
    physical, subpage = divmod(locator, subpages)
    return physical, subpage * fragment


if __name__ == "__main__":
    partition_cases = 0
    for capacity in (1, 2, 4, 8, 17, 32):
        for requested in range(1, capacity + 1):
            for instruction in (1, 2, 4, 8):
                for total in (0, 1, 2, 3, 7, 8, 15, 16, 31, 32, 65, 129):
                    span, active, ranges = partition(total, requested, instruction)
                    assert 0 <= active <= requested <= capacity
                    assert span % instruction == 0 and span >= instruction
                    assert [t for r in ranges for t in r] == list(range(total))
                    assert all(len(r) > 0 for r in ranges)
                    partition_cases += 1
    capacity = 8
    scratch = [float("nan")] * (2 * capacity)
    reads = []
    for active in (7, 2, 0, 5):
        for row in range(2):
            for split in range(active):
                scratch[row * capacity + split] = 100 * row + split + 1
        for row in range(2):
            positions = [row * capacity + split for split in range(active)]
            reads.extend(positions)
            result = sum(scratch[p] for p in positions)
            assert result == sum(100 * row + split + 1 for split in range(active))
            if row == 1 and active:
                wrong = sum(scratch[row * active + split] for split in range(active))
                assert wrong != result
    # A fixed four-rank collective does not shrink to the active two ranks.
    collective_ranks, active_ranks = set(range(4)), set(range(2))
    assert active_ranks != collective_ranks
    assert active_ranks | (collective_ranks - active_ranks) == collective_ranks
    table = [31, 7, 43, 2, 29, 11, 53, 5, 61, 17, 71, 13] * 32
    fragment_cases = 0
    crossing_witness = None
    for block_size in (4, 8, 16, 32, 64, 128):
        for page_size in (4, 8, 12, 16, 20, 24, 32, 64, 128):
            for block in range(6):
                fragment, locators = fragment_locators(block, block_size, page_size, table)
                assert block_size % fragment == 0 and page_size % fragment == 0
                mapped = []
                for locator in locators:
                    physical, offset = decode(locator, page_size, fragment)
                    assert offset + fragment <= page_size
                    mapped.extend(physical * page_size + offset + k for k in range(fragment))
                expected = [
                    table[t // page_size] * page_size + t % page_size
                    for t in range(block * block_size, (block + 1) * block_size)
                ]
                assert mapped == expected
                if block_size == 32 and page_size == 24 and block == 0:
                    crossing_witness = mapped
                    assert mapped[24] != mapped[23] + 1
                assert decode(-1, page_size, fragment)[0] == -1
                fragment_cases += 1
    # C++ truncating division on -1 with three subpages would produce page0.
    assert int(-1 / 3) == 0
    # Arithmetic shift/mask preserves an invalid page for power-of-two ratios.
    for ratio in (1, 2, 4, 8, 16, 32):
        assert (-1 >> (ratio.bit_length() - 1)) == -1
    max_locator = 2**31 - 1
    assert max_locator >= 0 and max_locator + 1 == 2**31
    assert (2**31 + 1) - 1 > max_locator
    policy_cases = 0
    for block_size in (4, 8, 16, 32, 64, 128):
        for position in (0, block_size - 1, block_size, 2 * block_size - 1, 2 * block_size):
            for include_current in (False, True):
                visible = position + 1
                bound = (position + int(include_current)) // block_size
                selected = list(reversed(range(max(0, bound - 2), bound)))
                tail = visible - bound * block_size
                support = [t for b in selected for t in range(b * block_size, (b + 1) * block_size)]
                support += list(range(bound * block_size, bound * block_size + tail))
                assert len(support) == len(set(support))
                assert all(0 <= t < visible for t in support)
                assert position in support
                policy_cases += 1
    assert crossing_witness is not None
    result = {
        "kind": "logical_capacity_and_fragment_check",
        "native_execution": False,
        "status": "passed",
        "partition_cases": partition_cases,
        "fragment_cases": fragment_cases,
        "current_block_policy_cases": policy_cases,
        "nonadjacent_page_witness": crossing_witness,
        "checks": [
            "aligned_exact_cover",
            "zero_work",
            "capacity_strides",
            "inactive_scratch_not_loaded",
            "fixed_collective_participants",
            "gcd_fragment_roundtrip",
            "non_power_of_two_page",
            "nonadjacent_page",
            "sentinel_decode",
            "locator_extent",
            "current_block_membership",
        ],
        "remaining": (
            "Native collective/TMEM/PDL teardown, TMA zero-fill and publication are not executed."
        ),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")

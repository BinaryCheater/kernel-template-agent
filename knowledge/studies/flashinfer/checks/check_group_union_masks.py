"Logical set-union masks and reusable staging; not device attention execution."

import json
from pathlib import Path

if __name__ == "__main__":
    rows = [[65, 1, 1, -1], [2, 65], [], [33], [2]]
    sets = [set(k for k in row if 0 <= k < 128) for row in rows]
    keys = sorted(set().union(*sets))
    slots = [[9, 3][k // 64] * 64 + k % 64 for k in keys]
    assert keys == [1, 2, 33, 65] and slots == [577, 578, 609, 193]
    masks = [sum(1 << i for i, k in enumerate(keys) if k in row) for row in sets]
    assert masks == [9, 10, 0, 4, 2]
    for expected, mask in zip(sets, masks, strict=True):
        assert {k for i, k in enumerate(keys) if mask >> i & 1} == expected
    # Teams0..3 process rows0..3, then team0 processes row4 with a cleared stage.
    stage0 = masks[0]
    stale = stage0 | masks[4]
    stage0 = 0
    stage0 |= masks[4]
    assert stale == 11 and stage0 == 2
    # Changing physical order without permuting masks loses key identity.
    physical_sort = sorted(range(len(keys)), key=lambda i: slots[i])
    wrong = {keys[i] for col, i in enumerate(physical_sort) if masks[3] >> col & 1}
    assert wrong != sets[3]
    values = {1: 0.0, 2: 3.0, 33: 6.0, 65: 9.0}
    masked = sum(values[k] for k in sets[3]) / len(sets[3])
    unmasked = sum(values[k] for k in keys) / len(keys)
    assert masked == 6 and unmasked == 4.5
    occurrence = (values[65] + values[1] + values[1]) / 3
    set_mean = (values[65] + values[1]) / 2
    assert occurrence == 3 and set_mean == 4.5
    result = {
        "kind": "logical_grouped_sparse_union_masks",
        "native_execution": False,
        "command": (
            ".venv/bin/python knowledge/studies/flashinfer/checks/check_group_union_masks.py"
        ),
        "logical_union": keys,
        "physical_slots": slots,
        "row_masks": masks,
        "stale_team_mask": stale,
        "cleared_team_mask": stage0,
        "row_masked_output": masked,
        "wrong_union_normalized_output": unmasked,
        "occurrence_output": occurrence,
        "set_output": set_mean,
        "remaining": (
            "Native bitmap atomics, scan/lane rank, mask staging publication a"
            "nd selected masked attention numeric contract"
        ),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")

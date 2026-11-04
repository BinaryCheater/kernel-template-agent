"Conditional channel and requantization witnesses; no native pipeline execution."

import json
import math
from pathlib import Path

if __name__ == "__main__":
    extra = [True, False, False, True, True, False, True, True]
    raw_generations = [0, 0]
    observations = []
    for tile, is_fp4 in enumerate(extra):
        slot = tile % 2
        if is_fp4:
            phase = raw_generations[slot] % 2
            observations.append(
                dict(tile=tile, slot=slot, raw_phase=phase, wrong_global_phase=(tile // 2) % 2)
            )
            raw_generations[slot] += 1
    assert observations[2] == dict(tile=4, slot=0, raw_phase=1, wrong_global_phase=0)
    assert raw_generations == [3, 2]
    # All four logical CLC rows map bijectively to native pair rank + unfolded query.
    fallback = [(q, m, m % 2, 2 * q + m // 2, 7) for q in range(2) for m in range(4)]
    native = {(rank, query, split) for _, _, rank, query, split in fallback}
    assert native == {(rank, query, 7) for query in range(4) for rank in range(2)}
    # Two finite E4M3 scale bytes for adjacent group16 inputs.
    small_scale, large_scale = 2**-9, 448.0
    small_value, large_value = 0.5 * small_scale, 6.0 * large_scale
    e = math.ceil(math.log2(max(large_value / 448.0, 1e-4)))
    common_scale = 2**e
    encoded_small = small_value / common_scale
    half_min_e4m3_subnormal = 2**-10
    assert common_scale == 8 and 0 < encoded_small < half_min_e4m3_subnormal
    result = {
        "kind": "logical_conditional_channels_fallback_rows_and_requantization",
        "native_execution": False,
        "command": (
            ".venv/bin/python knowledge/studies/flashinfer/checks/check_mixed_cache_channels.py"
        ),
        "raw_ready_events": observations,
        "raw_generations_per_slot": raw_generations,
        "kv_ready_generations_per_slot": [4, 4],
        "fallback_coordinates": fallback,
        "small_group16_value": small_value,
        "group32_power_two_scale": common_scale,
        "fp8_normalized_small_value": encoded_small,
        "nearest_even_e4m3_small_value_rounds_to_zero": True,
        "remaining": (
            "Native channel waits/arrivals and cluster branches, quantizer ins"
            "truction rounding, byte/fragment maps and full numerical error po"
            "licy"
        ),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")

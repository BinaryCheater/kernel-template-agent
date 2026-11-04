"Logical page/prefetch coordinates and common-frame value ownership; no CUDA run."

import json
import math
from pathlib import Path


def page_address(request, token, head, feature, layout):
    indptr, pages, page_size, heads, dimension = [0, 3, 8], [7, 1, 9, 6, 2, 8, 0, 5], 16, 2, 128
    page_iter, entry = divmod(indptr[request] * page_size + token, page_size)
    physical = pages[page_iter]
    base = physical * page_size * heads * dimension
    if layout == "NHD":
        return base + entry * heads * dimension + head * dimension + feature
    return base + head * page_size * dimension + entry * dimension + feature


def full_weighted_output(weights, values):
    mass = sum(weights)
    return [
        sum(w * row[d] for w, row in zip(weights, values, strict=True)) / mass
        for d in range(len(values[0]))
    ]


def feature_owned_stream(tiles, width):
    reference, mass = None, 0.0
    outputs = [[0.0] * (width // 4) for _ in range(4)]
    for weights, values in tiles:
        new_reference = max(
            [math.log(w) for w in weights] + ([] if reference is None else [reference])
        )
        correction = 0.0 if reference is None else math.exp(reference - new_reference)
        p = [math.exp(math.log(w) - new_reference) for w in weights]
        mass = mass * correction + sum(p)
        for owner in range(4):
            for local_d in range(width // 4):
                d = owner * (width // 4) + local_d
                outputs[owner][local_d] = outputs[owner][local_d] * correction + sum(
                    value * row[d] for value, row in zip(p, values, strict=True)
                )
        reference = new_reference
    return [value / mass for output in outputs for value in output]


if __name__ == "__main__":
    addresses = {layout: page_address(1, 35, 1, 61, layout) for layout in ["NHD", "HND"]}
    assert addresses == {"NHD": 33725, "HND": 35261}
    # All feature lanes read one token address; cooperative index lanes prefetch tokens.
    bdx, bdy, bdz, stages = 16, 4, 2, 2
    offsets = list(range(bdx * bdy * bdz))
    for iteration in range(34):
        if (iteration + stages) % bdx == 0:
            offsets = [(iteration + stages) * bdy * bdz + i for i in range(len(offsets))]
        for z in range(bdz):
            for y in range(bdy):
                index = (((iteration + stages) % bdx) * bdz + z) * bdy + y
                assert offsets[index] == (iteration + stages) * bdy * bdz + z * bdy + y
    assert page_address(1, 35, 1, 61, "NHD") != page_address(0, 35, 1, 61, "NHD")

    width = 512
    weights = [1.0, 2.0, 3.0, 4.0]
    values = [[100 * (key + 1) + d for d in range(width)] for key in range(4)]
    expected = full_weighted_output(weights, values)
    owned = feature_owned_stream([(weights, values)], width)
    assert all(math.isclose(x, 300 + d, abs_tol=1e-10) for d, x in enumerate(owned))
    assert all(math.isclose(x, y, abs_tol=1e-10) for x, y in zip(expected, owned, strict=True))
    wrong = [values[d // (width // 4)][d] for d in range(width)]
    assert wrong != owned
    more_weights = [8.0, 4.0, 2.0, 1.0]
    more_values = [[100 * (key + 5) + d for d in range(width)] for key in range(4)]
    streamed = feature_owned_stream([(weights, values), (more_weights, more_values)], width)
    direct = full_weighted_output(weights + more_weights, values + more_values)
    assert all(math.isclose(x, y, abs_tol=1e-10) for x, y in zip(direct, streamed, strict=True))

    # IEEE zero times a stale non-finite V is not a neutral contribution.
    assert math.isnan(0.0 * math.nan)
    assert 0.0 * 0.0 == 0.0
    result = {
        "kind": "logical_classic_attention_mapping_and_value_ownership",
        "native_execution": False,
        "command": (
            ".venv/bin/python knowledge/studies/flashinfer/checks/check_classic_attention.py"
        ),
        "page_address_witness": addresses,
        "offset_prefetch_iterations": 34,
        "prefetch_coordinates": "passed across two horizon refreshes",
        "value_features": width,
        "feature_owners": 4,
        "single_tile_weighted_output_first_last": [owned[0], owned[-1]],
        "two_tile_common_frame": "matches full key sum",
        "wrong_key_local_pv_changes_result": True,
        "zero_weight_stale_nan_is_not_zero": True,
        "remaining": (
            "Native P layout/publication, cp.async lifetimes and FP16/BF16 numeric accuracy"
        ),
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")

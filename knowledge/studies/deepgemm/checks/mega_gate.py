"Pinned MegaGate source-equation replays; no CUDA execution or RNG emulation."

from __future__ import annotations

import itertools
import json
import math
import random
import struct
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def sort4(values: list[float]) -> tuple[list[float], int]:
    entries = [(score, i) for i, score in enumerate(values)]
    for a, b in ((0, 1), (2, 3), (0, 2), (1, 3), (1, 2)):
        if entries[b][0] > entries[a][0] or (
            entries[b][0] == entries[a][0] and entries[b][1] < entries[a][1]
        ):
            entries[a], entries[b] = entries[b], entries[a]
    return [v for v, _ in entries], sum(i << (2 * j) for j, (_, i) in enumerate(entries))


def source_topk(values: list[float], k: int) -> list[int]:
    assert len(values) % 128 == 0
    waves = len(values) // 128
    runs = {}
    cursors = [0] * 32
    for lane, wave in itertools.product(range(32), range(waves)):
        base = wave * 128 + lane * 4
        runs[lane, wave] = sort4(values[base : base + 4])
    result = []
    for _ in range(k):
        offers = []
        for lane in range(32):
            best = (-math.inf, -1, 0)
            for wave in range(waves):
                cursor = (cursors[lane] >> (wave * 3)) & 7
                scores, perm = runs[lane, wave]
                score = -math.inf if cursor == 4 else scores[cursor]
                identity = wave * 128 + lane * 4 + ((perm >> (cursor * 2)) & 3)
                if score > best[0]:
                    best = (score, identity, wave)
            offers.append(best)
        maximum = max(o[0] for o in offers)
        identity = min(o[1] for o in offers if o[0] == maximum and o[1] >= 0)
        for lane, offer in enumerate(offers):
            if offer[1] == identity:
                cursors[lane] += 1 << (offer[2] * 3)
        result.append(identity)
    return result


def check_topk() -> dict[str, object]:
    network_cases = 0
    for values in itertools.product((-1.0, 0.0, 1.0), repeat=4):
        sorted_values, perm = sort4(list(values))
        offsets = [(perm >> (2 * i)) & 3 for i in range(4)]
        oracle = sorted(range(4), key=lambda i: (-values[i], i))
        assert offsets == oracle
        assert sorted_values == [values[i] for i in oracle]
        network_cases += 1
    rng, cases = random.Random(901), 0
    for experts, k, mode in itertools.product(
        (4, 64, 72, 108, 128, 260, 264, 384, 512), (1, 4, 8, 16, 32), range(4)
    ):
        if k > experts:
            continue
        valid = [float(rng.randrange(-8, 9)) for _ in range(experts)]
        if mode == 0:
            valid = [0.0] * experts
        elif mode == 1:
            valid = list(map(float, range(experts)))
        elif mode == 2:
            valid = list(map(float, reversed(range(experts))))
        padded = valid + [-math.inf] * ((experts + 127) // 128 * 128 - experts)
        expected = sorted(range(experts), key=lambda i: (-valid[i], i))[:k]
        assert source_topk(padded, k) == expected
        cases += 1
    return {"sorting_network_cases": network_cases, "full_selection_domains": cases}


def check_routing() -> dict[str, object]:
    scores, bias = [1.0, 2.0, 4.0, 8.0], [10.0, 0.0, 0.0, -20.0]
    ids = sorted(range(4), key=lambda i: (-(scores[i] + bias[i]), i))[:2]
    assert ids == [0, 2]
    weights = [scores[i] / sum(scores[j] for j in ids) * 1.5 for i in ids]
    assert weights == [0.30000000000000004, 1.2000000000000002]
    wrong = [(scores[i] + bias[i]) / sum(scores[j] + bias[j] for j in ids) * 1.5 for i in ids]
    assert weights != wrong
    # Fixed routing retains slot multiplicity; it is not the unique top-k contract.
    fixed = [2, 2, 0]
    assert [scores[i] / sum(scores[j] for j in fixed) for i in fixed] == [4 / 9, 4 / 9, 1 / 9]
    raw = [-200.0, -199.0]
    rounded_exp = [struct.unpack("f", struct.pack("f", math.exp(x)))[0] for x in raw]
    activated = [math.sqrt(math.log1p(x)) for x in rounded_exp]
    assert activated == [0.0, 0.0]
    assert max(range(2), key=lambda i: raw[i]) == 1
    assert sorted(range(2), key=lambda i: (-activated[i], i))[0] == 0
    map_domains = 0
    for rank, token, count in itertools.product(
        (0, 1, 7), (0, 1, 17, (1 << 20) - 1), (1, 2, 3, 33)
    ):
        duplicate = ((rank + token * 23333) & 0xFFFFFFFF) % count
        assert 0 <= duplicate < count
        map_domains += 1
    # Source uint32 arithmetic differs from an unbounded integer at large token indices.
    token = (1 << 20) - 1
    assert ((token * 23333) & 0xFFFFFFFF) % 3 != (token * 23333) % 3
    return {
        "bias_example": {"selected": ids, "unbiased_weights": weights},
        "fixed_duplicate_slots": fixed,
        "rounded_activation_tie": {
            "raw": raw,
            "activated": activated,
            "raw_winner": 1,
            "activated_tie_winner": 0,
        },
        "physical_duplicate_domains": map_domains,
        "random_mapping_premise": (
            "Random routing emits dense physical IDs, not to_physical_map values."
        ),
    }


def check_random_distinct() -> int:
    cases = 0
    for n in range(1, 8):
        for k in range(1, min(n, 4) + 1):
            for draws in itertools.product(*(range(n - i) for i in range(k))):
                candidates, selected = list(draws), [-1] * k
                for _ in range(k):
                    value, lane = min((v, i) for i, v in enumerate(candidates) if v is not None)
                    for i, candidate in enumerate(candidates):
                        if candidate is None:
                            continue
                        if i == lane:
                            selected[i], candidates[i] = candidate, None
                        elif candidate >= value and lane < i:
                            candidates[i] += 1
                assert len(set(selected)) == k and all(0 <= i < n for i in selected)
                cases += 1
    return cases


def check_schedule() -> int:
    domains = 0
    for blocks, groups, logical, warps, width in itertools.product(
        (1, 2, 5, 17), (1, 3, 8), (1, 2, 4, 8, 16), (4, 12, 28), (16, 48, 256)
    ):
        owners = []
        for group in range(min(groups, blocks)):
            for block in range(group, blocks, min(groups, blocks)):
                tokens = []
                for cta, warp in itertools.product(range(logical), range(warps)):
                    tokens.extend(range(warp * logical + cta, width, warps * logical))
                assert sorted(tokens) == list(range(width))
                owners.append(block)
        assert sorted(owners) == list(range(blocks))
        domains += 1
    # Input ring position carries across token blocks, independent of the two-slot accumulator ring.
    stages, kblocks = 5, 8
    carried = [(i * kblocks) % stages for i in range(4)]
    assert carried == [0, 3, 1, 4] and carried != [0] * 4
    return domains


def main() -> None:
    root = Path(json.loads((STUDY / "inputs/sources.json").read_text())["tree"])
    epilogue = (root / "deep_gemm/include/deep_gemm/epilogue/sm100_mega_gate.cuh").read_text()
    device = (root / "deep_gemm/include/deep_gemm/impls/sm100_bf16_mega_gate.cuh").read_text()
    for marker in ("cursor == 4", "expert_wave_idx * 3", "global_token_idx * 23333u", "1e-20f"):
        assert marker in epilogue
    for marker in (
        "kHasScoringBias and kNumSplitK == 1",
        "split_idx = 1",
        "is_force_random",
        "has_fixed_routing",
    ):
        assert marker in device
    assert (
        device.index("if (is_masked)")
        < device.index("if (is_force_random)")
        < device.index("if (has_fixed_routing)")
    )
    result = {
        "outcome": "PASS",
        "topk": check_topk(),
        "routing": check_routing(),
        "random_distinct_draw_domains": check_random_distinct(),
        "token_schedule_domains": check_schedule(),
        "boundary": (
            "Scalar source equations; no CUDA, collective, curand distribution"
            ", TF32 or synchronization execution."
        ),
    }
    (STUDY / "checks/mega-gate-result.json").write_text(json.dumps(result, indent=2) + "\n")
    sys.stdout.write(
        "PASS: sorted-run top-k, routing boundaries, distinct draws and token ownership.\n"
    )


if __name__ == "__main__":
    main()

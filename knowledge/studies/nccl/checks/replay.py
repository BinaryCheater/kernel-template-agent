"""Source-study algebra checks; no network or CUDA execution."""

import json
from pathlib import Path


def byte_pack(xs):
    return sum(x << (8 * i) for i, x in enumerate(xs))


def main():
    mask = 0x00FF00FF
    for a in range(256):
        for b in range(256):
            aa = byte_pack([a, 255 - a, a ^ 85, 255])
            bb = byte_pack([b, b ^ 170, 255 - b, 1])
            x = (aa & mask) + (bb & mask)
            y = ((aa & ~mask) + (bb & ~mask)) & 0xFFFFFFFF
            got = (x & mask) | (y & (~mask & 0xFFFFFFFF))
            expected = byte_pack([((aa >> (8 * i)) + (bb >> (8 * i))) & 255 for i in range(4)])
            assert got == expected
    assert (byte_pack([255, 0, 0, 0]) + byte_pack([1, 0, 0, 0])) != byte_pack([0, 0, 0, 0])
    for heads in range(1, 8):
        for channels in range(1, 6):
            ordinary = [c * heads + h for c in range(channels) for h in range(heads)]
            registered = [h * channels + c for c in range(channels) for h in range(heads)]
            assert sorted(ordinary) == sorted(registered) == list(range(heads * channels))
    for peers in range(1, 10):
        for b in range(1, 6):
            B = 1 << b
            credits = [B // peers + (step < B % peers) for step in range(peers)]
            assert sum(credits) == B
            for step in range(peers):
                assert (step + B) % peers == (step + B % peers) % peers
    # Equal coordinate mapping, disjoint owners and complete peer fan-in.
    P, blocks, warps = 3, 2, 2
    ids = [
        lane + 32 * (rank + P * (block + blocks * warp))
        for lane in range(32)
        for rank in range(P)
        for block in range(blocks)
        for warp in range(warps)
    ]
    assert sorted(ids) == list(range(32 * P * blocks * warps))
    inputs = [[100 * r + i for i in range(19)] for r in range(P)]
    result = [
        sum(inputs[(owner + j) % P][i] for j in range(P)) for i in range(19) for owner in [i % P]
    ]
    assert result == [300 + P * i for i in range(19)]
    # Model subtree contribution sets; duplicate rank would violate disjointness.
    children = {0: [1, 2], 1: [3, 4], 2: [5], 3: [], 4: [], 5: []}

    def represented(node):
        sets = [represented(c) for c in children[node]]
        union = {node}
        for s in sets:
            assert union.isdisjoint(s)
            union |= s
        return union

    assert represented(0) == set(range(6))
    Path(__file__).with_name("replay-result.json").write_text(
        json.dumps(
            {
                "outcome": "passed",
                "checks": [
                    "65536 packed-byte paired input patterns",
                    "wide-add carry counterexample",
                    "head/channel address permutations",
                    "credit totals and alias-next-peer map",
                    "flattened peer owner bijection and fan-in",
                    "tree disjoint subtree cover",
                ],
                "scope": (
                    "local integer algebra only; no peer visibility, GPU execution, pr"
                    "ovider ordering or ProofResult"
                ),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

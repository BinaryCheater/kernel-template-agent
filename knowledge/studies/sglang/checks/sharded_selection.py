"""Rank bootstrap and bounded candidate retry; no native protocol validation."""

import json
from itertools import permutations, product
from pathlib import Path

cases = 0
for n in range(2, 9):
    for values in product(range(3), repeat=n):
        records = list(enumerate(values))
        for workers in (2, 3):
            shards = [records[w::workers] for w in range(workers)]
            bootstrap = [r for shard in shards for r in shard[:1]]
            for k in range(1, min(n, len(bootstrap)) + 1):
                winners = sorted(bootstrap, key=lambda r: (-r[1], r[0]))[:k]
                witness_ids = {i for i, v in winners}
                b = min(v for i, v in winners)
                candidates = []
                for shard in shards:
                    local = [r for r in shard[:1] if r[0] in witness_ids]
                    threshold = b
                    cap = max(k, 2)
                    for off in range(1, len(shard), 2):
                        stage = shard[off : off + 2]
                        hits = [r for r in stage if r[1] > threshold]
                        if len(local) + len(hits) <= cap:
                            local += hits
                        else:
                            local = sorted(local + stage, key=lambda r: (-r[1], r[0]))[:k]
                            assert len(local) == k
                            raised = min(v for i, v in local)
                            assert raised >= threshold
                            threshold = raised
                    candidates += local
                got = sorted((v for i, v in candidates), reverse=True)[:k]
                assert got == sorted(values, reverse=True)[:k]
                cases += 1
reservation_cases = 0
for order in permutations((0, 1, 2)):
    for counts in product(range(4), repeat=3):
        for cap in range(2, 9):
            committed = [("old", 0), ("old", 1)]
            counter = len(committed)
            storage = committed + [None] * (cap - len(committed))
            all_fit = True
            for worker in order:
                start = counter
                counter += counts[worker]
                fits = counter <= cap
                all_fit &= fits
                if fits:
                    for j in range(counts[worker]):
                        storage[start + j] = (worker, j)
            assert storage[:2] == committed
            if not all_fit:
                counter -= sum(counts)
                assert counter == len(committed)
                # Suffix writes become dead. Original stage records supply retry input.
                retry = [(worker, j) for worker in range(3) for j in range(counts[worker])]
                assert len(set(retry)) == sum(counts)
            else:
                assert counter == 2 + sum(counts)
            reservation_cases += 1
Path(__file__).with_suffix(".json").write_text(
    json.dumps(
        {
            "passed": True,
            "rank_cases": cases,
            "reservation_cases": reservation_cases,
            "scope": (
                "Exact three-valued sharded rank witnesses and all sequential clai"
                "m orders for bounded counters; no GPU ordering, timing or livenes"
                "s proof"
            ),
        },
        indent=2,
    )
    + "\n"
)

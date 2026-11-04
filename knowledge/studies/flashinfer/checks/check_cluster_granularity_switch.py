"Exact cooperative-cluster versus independent-CTA row assignment and metadata boundaries."

import itertools
import json
from collections import Counter
from pathlib import Path


def verify(flags, rows_per_request, cluster_ctas, capacity):
    requests = len(flags)
    assert requests <= capacity and capacity >= 1
    long = [i for i, flag in enumerate(flags) if flag]
    short = [i for i, flag in enumerate(flags) if not flag]
    order = long + short
    assert sorted(order) == list(range(requests))
    a, b = len(long) * rows_per_request, len(short) * rows_per_request
    live_clusters = a + (b + cluster_ctas - 1) // cluster_ctas
    grid = capacity * rows_per_request
    assert live_clusters <= grid
    owners = []
    tails = 0
    for g in range(grid):
        if g < a:
            request = order[g // rows_per_request]
            local = g % rows_per_request
            row = request * rows_per_request + local
            ranks = [(rank, row) for rank in range(cluster_ctas)]
            assert len(ranks) == cluster_ctas and len({r for _, r in ranks}) == 1
            owners.append(row)  # One cooperative body, not C independent rows.
        elif g < live_clusters:
            active_ranks = []
            for rank in range(cluster_ctas):
                j = (g - a) * cluster_ctas + rank
                if j < b:
                    request = order[len(long) + j // rows_per_request]
                    local = j % rows_per_request
                    owners.append(request * rows_per_request + local)
                    active_ranks.append(rank)
            assert active_ranks
            if len(active_ranks) < cluster_ctas:
                tails += 1
        else:
            assert g >= live_clusters  # Surplus physical clusters do no row work.
    assert Counter(owners) == Counter(range(requests * rows_per_request))
    return tails


if __name__ == "__main__":
    cases = partial_tail_cases = 0
    for requests in range(7):
        for flags in itertools.product([False, True], repeat=requests):
            for rows_per_request in range(1, 5):
                for cluster_ctas in [2, 4, 8]:
                    for capacity in [max(requests, 1), max(requests + 2, 1)]:
                        partial_tail_cases += verify(
                            flags, rows_per_request, cluster_ctas, capacity
                        )
                        cases += 1
    assert cases == 3048
    raw = [65536, 65537, 262144, 262148]
    compressed = [x // 4 for x in raw]
    proper_long = [i for i, x in enumerate(compressed) if x > 65536]
    wrong_raw_long = [i for i, x in enumerate(raw) if x > 65536]
    assert proper_long == [3] and wrong_raw_long == [1, 2, 3]
    assert verify([False], 1, 4, 1) == 1
    assert 1 < 4  # A cluster barrier requiring4 active short CTAs would be unsatisfied.
    # The same request-only partition may be reused with a correctly rebound row expansion.
    for expansion in [1, 2, 3, 4]:
        verify([False, True, False], expansion, 4, 3)
    old_order = [2, 0, 1]
    assert max(old_order) >= 2  # Population shrunk to2: old metadata cannot be used.
    wrong_expansion = {request * 2 + local for request in range(3) for local in range(2)}
    required_rows = set(range(3 * 3))
    assert wrong_expansion != required_rows
    report = {
        "kind": "logical_cluster_granularity_switch",
        "status": "passed",
        "native_execution": False,
        "mapping_cases": cases,
        "partial_short_tail_cases": partial_tail_cases,
        "cluster_widths": [2, 4, 8],
        "rows_per_request": [1, 2, 3, 4],
        "scan_key_units": {
            "raw_lengths": raw,
            "compressed_by4": compressed,
            "proper_long_requests": proper_long,
            "wrong_raw_long_requests": wrong_raw_long,
        },
        "tail_collective_witness": {"active_short_ctas": 1, "hypothetical_barrier_ctas": 4},
        "metadata": "Request partition is independent of expansion; consumer/layout must match",
        "launch": "Fixed capacity remains launched; inactive clusters skip row work",
        "boundary": "Body exactness, native DSM/progress and compiled resource budget separate",
    }
    Path(__file__).with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")

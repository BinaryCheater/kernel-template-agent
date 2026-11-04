"""Bounded synchronous-wave capacity and padded localized-address checks."""

from __future__ import annotations

import itertools
import json
import math
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]


def div(n: int, d: int) -> int:
    return (n + d - 1) // d


def policy(blocks: int, clusters: int, n1: int, n2: int) -> tuple[int, int]:
    warm = max(
        div(div(clusters, n2) * n1, clusters),
        div(n1 + (blocks - 1) * max(n1 - n2, 0), clusters) + 1,
    )
    warm = min(warm, div(blocks * n1, clusters))
    started = div(min(warm * clusters, blocks * n1), n1)
    growth = div(blocks * max(n2 - n1, 0), n2)
    bound = min(blocks, started + growth + div(clusters, min(n1, n2)))
    return warm, bound


def simulate_waves(blocks: int, clusters: int, n1: int, n2: int, warm: int) -> int:
    issued1, issued2, live, peak = 0, 0, set(), 0

    def wave(layer: int) -> None:
        nonlocal issued1, issued2, peak
        count = min(
            clusters, blocks * (n1 if layer == 1 else n2) - (issued1 if layer == 1 else issued2)
        )
        for _ in range(count):
            if layer == 1:
                block = issued1 // n1
                live.add(block)
                issued1 += 1
                peak = max(peak, len(live))
            else:
                block = issued2 // n2
                assert issued1 >= (block + 1) * n1
                issued2 += 1
                if issued2 % n2 == 0:
                    live.remove(block)

    for _ in range(warm):
        wave(1)
    while issued2 < blocks * n2:
        wave(2)
        wave(1)
    assert issued1 == blocks * n1 and not live
    return peak


def check_wave_bound() -> dict[str, object]:
    cases, tight = 0, 0
    for blocks, clusters, n1, n2 in itertools.product(
        (0, 1, 2, 3, 7, 16, 33, 129), (1, 2, 4, 8, 66), (1, 2, 4, 8), (1, 2, 4, 8)
    ):
        warm, bound = policy(blocks, clusters, n1, n2)
        peak = simulate_waves(blocks, clusters, n1, n2, warm)
        assert peak <= bound <= blocks
        tight += peak == bound
        cases += 1
    return {
        "synchronous_wave_domains": cases,
        "bound_tight_domains": tight,
        "boundary": "No asynchronous skew, stealing, TMA latency or metadata lookahead.",
    }


def check_localized() -> dict[str, int]:
    coordinates, domains = 0, 0
    for batch, n, k, elem, gran in itertools.product((1, 3), (4, 10), (8, 24), (1, 2), (64, 256)):
        per = n // 2
        slice_bytes = batch * per * k * elem
        padded = div(slice_bytes, gran) * gran
        tma_strides = (k * elem, padded, per * k * elem)
        seen = set()
        for b, row, col in itertools.product(range(batch), range(n), range(k)):
            domain, local_row = divmod(row, per)
            descriptor = (
                col * elem
                + local_row * tma_strides[0]
                + domain * tma_strides[1]
                + b * tma_strides[2]
            )
            oracle = domain * padded + ((b * per + local_row) * k + col) * elem
            assert descriptor == oracle and descriptor not in seen
            seen.add(descriptor)
            coordinates += 1
        assert padded % gran == 0 and 0 <= padded - slice_bytes < gran
        domains += 1
    assert math.gcd(7, 32) == 1
    assert len({(i * 7) % 32 for i in range(32)}) == 32
    return {
        "padded_descriptor_domains": domains,
        "tagged_coordinates": coordinates,
        "probe_chain_lines": 32,
    }


def main() -> None:
    root = Path(json.loads((STUDY / "inputs/sources.json").read_text())["tree"])
    scheduler = (root / "deep_gemm/include/deep_gemm/scheduler/mega_moe.cuh").read_text()
    assert "num_live_blocks_after_warmup + frontier_growth + wave_margin" in scheduler
    ipc = (root / "deep_gemm/utils/mlopart.py").read_text()
    for marker in (
        "REQUEST = struct.Struct('qq')",
        "socket.send_fds",
        "self.client.wait()",
        "check(cuda.cuMemRelease(handle))",
    ):
        assert marker in ipc
    result = {
        "outcome": "PASS",
        "capacity": check_wave_bound(),
        "localized": check_localized(),
        "boundary": (
            "Source equation replay only; no allocator, IPC, probe or GPU scheduling execution."
        ),
    }
    (STUDY / "checks/moe-capacity-localized-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    sys.stdout.write("PASS: synchronous-wave capacity and padded localized descriptor addresses.\n")


if __name__ == "__main__":
    main()

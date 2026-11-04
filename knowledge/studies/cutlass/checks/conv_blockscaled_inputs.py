"Replay mixed completion counters, shifted arrivals and convolution scale indices."

from __future__ import annotations

import ast
import hashlib
import itertools
import json
import math
import sys
import textwrap
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
SOURCE = (
    "examples/python/CuTeDSL/cute/blackwell/kernel/conv/dense_blockscaled_implicit_gemm_fprop.py"
)


def interleavings(copy_count: int, chunks: int) -> int:
    # State: registered, completed transaction chunks, completed copies, arrived copies.
    todo = [(False, 0, 0, 0)]
    seen = set(todo)
    while todo:
        registered, tx, done, arrived = todo.pop()
        bytes_pending = chunks - tx if registered else 0
        arrivals_pending = copy_count + 1 - arrived - int(registered)
        full = bytes_pending == 0 and arrivals_pending == 0
        assert full == (registered and tx == chunks and arrived == copy_count)
        assert not full or done == copy_count
        next_states = []
        if not registered:
            next_states.append((True, tx, done, arrived))
        if registered and tx < chunks:
            next_states.append((registered, tx + 1, done, arrived))
        if done < copy_count:
            next_states.append((registered, tx, done + 1, arrived))
        if arrived < done:
            next_states.append((registered, tx, done, arrived + 1))
        for state in next_states:
            if state not in seen:
                seen.add(state)
                todo.append(state)
    return len(seen)


def shifted_arrivals(tiles: int, slots: int) -> tuple[bool, list[int]]:
    pending = []
    active_slots = {}
    signaled = []
    for tile in range(tiles):
        slot = tile % slots
        if slot in active_slots:
            return False, signaled
        active_slots[slot] = tile
        pending.append(tile)
        if tile >= 2:
            old = pending.pop(0)
            signaled.append(old)
            del active_slots[old % slots]
    for old in pending:
        signaled.append(old)
        del active_slots[old % slots]
    assert not active_slots
    return True, signaled


def main() -> None:
    manifest = json.loads((STUDY / "inputs/sources.json").read_text())
    root = Path(manifest["tree"])
    data = (root / SOURCE).read_bytes()
    retained = next(x for x in manifest["files"] if x["path"] == SOURCE)
    assert hashlib.sha256(data).hexdigest() == retained["sha256"]
    source = data.decode()
    tree = ast.parse(source)
    assert "sfa_cpasync_arrive_count = 257 if use_2cta_instrs else 129" in source
    assert "sfa_shift_K = 2" in source
    assert "producer_cp_async_wait(sfa_shift_K)" in source
    assert "producer_cp_async_wait(0)" in source
    funcs = [
        x
        for x in tree.body
        if isinstance(x, ast.FunctionDef)
        and x.name in {"sf_k_tile_channels", "sfb_per_position_channels"}
    ]
    ns: dict = {}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), SOURCE, "exec"), ns)
    recurrence = compile(
        textwrap.dedent("\n".join(source.splitlines()[2497:2519])), SOURCE + ":2498", "exec"
    )
    states = sum(
        interleavings(copies, chunks) for copies, chunks in itertools.product(range(1, 5), repeat=2)
    )
    # Without the reserved registration arrival, copy producers can publish before TMA exists.
    without_reserve = {"arrivals": 128, "bytes": 0, "registered": False}
    for _ in range(128):
        without_reserve["arrivals"] -= 1
    assert without_reserve["arrivals"] == without_reserve["bytes"] == 0
    assert not without_reserve["registered"]
    premature = {"registered": False, "byte_counter": 0, "arrival_counter_without_reservation": 0}
    delay_domains = 0
    for tiles, slots in itertools.product(range(33), range(1, 9)):
        progress, arrivals = shifted_arrivals(tiles, slots)
        assert progress == (slots >= min(tiles, 3))
        if progress:
            assert arrivals == list(range(tiles))
        delay_domains += 1
    assert not shifted_arrivals(3, 2)[0]
    assert shifted_arrivals(2, 2) == (True, [0, 1])
    assert shifted_arrivals(1, 1) == (True, [0])
    # One registration owner plus128 writes-complete SFA arrivals per cooperating CTA.
    for peers in (1, 2):
        assert peers * 128 + 1 == (129 if peers == 1 else 257)
    for rank in range(16):
        leader = rank & ~1
        assert leader == rank - rank % 2
        assert (leader ^ 1) & ~1 == leader

    index_domains = positions = 0
    wrong_unpadded_position = None
    for vec, tile_k in ((16, 64), (16, 128), (16, 256), (32, 64), (32, 128)):
        sf_span = ns["sf_k_tile_channels"](tile_k, vec)
        cadence = sf_span // tile_k
        cp_groups = sf_span // (vec * 4)
        for channels, taps in itertools.product(
            (4 * vec, 8 * vec, 12 * vec), ((1, 1, 1), (2, 2, 3))
        ):
            sf_extent = channels // vec
            c_tiles = math.ceil(channels / tile_k)
            sf_c_tiles = math.ceil(c_tiles / cadence)
            padded_c = ns["sfb_per_position_channels"](channels, tile_k, vec)
            assert padded_c == sf_c_tiles * sf_span
            T, R, S = taps
            env = {
                "sfa_s_idx": 0,
                "sfa_r_idx": 0,
                "sfa_t_idx": 0,
                "sfa_c_tile_idx": 0,
                "sfa_delta": 0,
                "sfa_w_in": 0,
                "sfa_h_in": 0,
                "sfa_d_in": 0,
                "w_in_base": 0,
                "h_in_base": 0,
                "d_in_base": 0,
                "conv_S": S,
                "conv_R": R,
                "conv_T": T,
                "c_tiles_per_fpos": c_tiles,
                "dil_w": 2,
                "dil_h": 1,
                "dil_d": 1,
                "sfa_step_s": 2 * sf_extent,
                "sfa_step_r": (7 - S * 2) * sf_extent,
                "sfa_step_t": (35 - R * 7) * sf_extent,
            }
            visits = []
            for iteration in range(c_tiles * S * R * T):
                ct, fpos = divmod(iteration, S * R * T)
                high, s = divmod(fpos, S)
                t, r = divmod(high, R)
                assert (
                    env["sfa_c_tile_idx"],
                    env["sfa_t_idx"],
                    env["sfa_r_idx"],
                    env["sfa_s_idx"],
                ) == (ct, t, r, s)
                assert env["sfa_delta"] == (t * 35 + r * 7 + s * 2) * sf_extent
                flat_sfb = ct // cadence + (s + S * (r + R * t)) * sf_c_tiles
                assert flat_sfb == (ct * tile_k // sf_span) + fpos * (padded_c // sf_span)
                for thread in range(128):
                    row = 8 * (thread // 32) + 32 * ((thread % 32) // 8) + thread % 8
                    assert 0 <= row < 128
                    seen = set()
                    for i in range(cp_groups):
                        si = ((thread % 32) // 8 + i) % cp_groups
                        seen.add(si)
                        start = min((ct // cadence) * (sf_span // vec) + si * 4, sf_extent - 4)
                        assert start % 4 == 0 and 0 <= start <= sf_extent - 4
                    assert seen == set(range(cp_groups))
                visits.append((ct, t, r, s))
                exec(recurrence, env)
                positions += 1
            assert set(visits) == set(
                itertools.product(range(c_tiles), range(T), range(R), range(S))
            )
            if padded_c != channels and S * R * T > 1:
                # A second filter position needs a fresh SF tile;192 channels do not fill256.
                wrong_unpadded_position = {
                    "channels": channels,
                    "tile_span": sf_span,
                    "correct_position_start": padded_c,
                    "wrong_position_start": channels,
                }
                assert padded_c // sf_span > channels // sf_span
            index_domains += 1
    assert wrong_unpadded_position is not None
    row_image = [8 * (tid // 32) + 32 * ((tid % 32) // 8) + tid % 8 for tid in range(128)]
    assert sorted(row_image) == list(range(128))
    result = {
        "outcome": "PASS",
        "counter_interleaving_states": states,
        "reserved_arrival_counterexample": premature,
        "delay_domains": delay_domains,
        "delay_capacity_counterexample": {"tiles": 3, "slots": 2, "progress": False},
        "scale_index_domains": index_domains,
        "source_recurrence_positions": positions,
        "wrong_unpadded_filter_position": wrong_unpadded_position,
        "boundary": (
            "Scalar event and retained integer recurrence models; target count"
            "er atomicity, visibility and multicast incidence remain open."
        ),
    }
    (STUDY / "checks/conv-blockscaled-inputs-result.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    sys.stdout.write("PASS: byte/arrival join, shifted tails and source SFA/SFB recurrence.\n")


if __name__ == "__main__":
    main()

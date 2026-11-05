"""PERFMODEL: the one L2 model (gb10sim/l2.py)."""

from __future__ import annotations

from kta.models.gb10sim.l2 import Access, Cache, Load, replay_walk, warm_miss


def _load(pc: int, offs: tuple, gx: int, shift: int) -> Load:
    return Load(
        0, pc, 0, -1, 4, ((("c", 0, pc), 1),), offs,
        shift=[{v: v * shift for v in range(gx)}, {0: 0}, {0: 0}],
    )  # fmt: skip


def test_cache_evicts_single_use_entries_first() -> None:
    c = Cache(2.0)
    assert c.touch("a", 1.0) and c.touch("b", 1.0)
    assert not c.touch("a", 1.0)  # a is now protected
    assert c.touch("c", 1.0)  # evicts b, not a
    assert not c.touch("a", 1.0)
    assert c.touch("b", 1.0)


def test_reuse_shows_per_load_instruction() -> None:
    # 8 blocks: a row per block (pc 1) and one vector every block shares (pc 2)
    offs = tuple(i * 128 for i in range(32))
    acc = Access(1, (8, 1, 1), [_load(1, offs, 8, 1 << 20), _load(2, offs, 8, 0)])
    r = replay_walk(acc, 8, 1e9)
    assert r.reads == (8 + 1) * 32 * 32  # the vector is read once, a miss brings a sector
    assert r.miss_by_pc == {1: 1.0, 2: 1 / 8}
    assert r.footprint == r.reads


def test_one_unknown_execution_is_a_part_of_its_load() -> None:
    # round 18: GEMV's last pass, guarded per lane: seven known executions of pc 1, one unknown
    known = [
        Load(
            0,
            1,
            i,
            -1,
            4,
            ((("c", 0, 1), 1),),
            tuple(j * 32 + i << 16 for j in range(32)),
            shift=[{0: 0}, {0: 0}, {0: 0}],
        )
        for i in range(7)
    ]
    lost = Load(0, 1, 7, -1, 4, None, None, shift=[{0: None}, {0: 0}, {0: 0}])
    r = replay_walk(Access(1, (1, 1, 1), [*known, lost]), 1, 1e9)
    assert r.loose_share[1] == 1 / 8


def test_warm_curve_is_flat_below_capacity_and_high_above() -> None:
    assert warm_miss(8 * 2**20) < 0.05 < 0.8 < warm_miss(40 * 2**20)


def test_prmt_sign_and_zero_extension() -> None:
    from kta.models.gb10sim.lanes import prmt

    assert prmt(0x0000_8001, 0x9910, 0) == 0xFFFF_8001  # 16-bit sign extension, division by 3
    assert prmt(0x0000_7001, 0x9910, 0) == 0x0000_7001
    assert prmt(0x1234_8001, 0x7710, 0) == 0x0000_8001  # zero extension
    assert prmt(0x0102_0304, 0x3210, 0) == 0x0102_0304


def test_lanes_follow_lea_hi_and_an_or_over_a_block_product() -> None:
    """Round 14: an index divided by a power of two, and `tid | block * k`, stay known addresses."""
    from kta.models.gb10sim.lanes import Lanes

    la = Lanes(0, (32, 1, 1))
    la.v["R0"] = ((), tuple(range(32)))
    la.v["R7"] = ((), (5,) * 32)
    la.step("LEA", "LEA.HI", ("R4", "R0", "R7", "RZ", "0x1e"), {}, True)
    assert la.v["R4"][1][:6] == (5, 5, 5, 5, 6, 6)
    la.v["R1"] = (((("b",), 0x140),), (0,) * 32)
    la.v["R2"] = ((), tuple(i >> 2 for i in range(32)))
    la.step("LOP3", "LOP3.LUT", ("R3", "R2", "R1", "RZ", "0xfc", "!PT"), {}, True)
    assert la.v["R3"][0] == la.v["R1"][0] and la.v["R3"][1][4] == 1


def test_real_pairs_split_by_the_passes() -> None:
    """Round 14: a wrong pair counts as real only if its measured order holds in every pass."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parents[2] / "tools"))
    from kta_perfmodel.blind import real_pairs

    g = [
        {"id": "a", "measured": 10.0, "predicted": 12.0},
        {"id": "b", "measured": 11.0, "predicted": 11.0},
        {"id": "c", "measured": 13.0, "predicted": 14.0},
    ]
    t = {"a": [10.0, 10.0], "b": [11.0, 9.0], "c": [13.0, 13.0]}
    c = real_pairs(g, t)
    assert (c["of"], c["right"], c["wrong_flips"], c["wrong_real"]) == (3, 2, 1, 0)
    assert (c["real_of"], c["real_right"]) == (2, 2)

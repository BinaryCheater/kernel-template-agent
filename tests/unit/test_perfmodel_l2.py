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


def test_warm_curve_is_flat_below_capacity_and_high_above() -> None:
    assert warm_miss(8 * 2**20) < 0.05 < 0.8 < warm_miss(40 * 2**20)

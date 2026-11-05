"""The lanes as a class of integer instructions (round 15): divisions, pairs, bitwise."""

import json

import pytest

from kta.models.gb10sim.lanes import Lanes


def lanes() -> Lanes:
    return Lanes(0, (1024, 1, 1))


def run(ls: Lanes, *code: tuple) -> None:
    for base, op, args in code:
        ls.step(base, op, args, {}, True)


def tid(ls: Lanes) -> None:
    run(ls, ("S2R", "S2R", ("R0", "SR_TID.X")))


def test_division_by_three_reads_the_high_half_of_a_wide_product() -> None:
    ls = lanes()
    tid(ls)
    run(
        ls,
        ("IMAD", "IMAD.WIDE.U32", ("R2", "R0", "-0x55555555", "RZ")),
        ("SHF", "SHF.R.U32.HI", ("R4", "RZ", "0x1", "R3")),
    )
    assert ls.get("R4", {}) == ((), tuple(i // 3 for i in range(32)))


def test_division_by_a_run_time_count_through_the_float_steps() -> None:
    ls = lanes()
    tid(ls)
    run(
        ls,
        ("MOV", "MOV", ("R8", "0x7")),
        ("I2F", "I2F.U32.RP", ("R6", "R8")),
        ("MUFU", "MUFU.RCP", ("R6", "R6")),
        ("IADD", "IADD", ("R4", "R6", "0xffffffe")),
        ("F2I", "F2I.FTZ.U32.TRUNC.NTZ", ("R5", "R4")),
    )
    q = ls.get("R5", {})
    assert q is not None and q[0] == () and 0x24924000 < q[1][0] < 0x24925000


def test_constants_of_the_compiler() -> None:
    ls = lanes()
    run(
        ls,
        ("CS2R", "CS2R", ("R1", "SRZ")),
        ("HFMA2", "HFMA2", ("R2", "-RZ", "RZ", "0", "2.384185791015625e-07")),
    )
    assert ls.get("R1", {}) == ((), (0,) * 32)
    assert ls.get("R2", {}) == ((), (4,) * 32)


def test_bitwise_over_a_base_uses_the_bits_it_leaves_free() -> None:
    ls = lanes()
    tid(ls)
    run(
        ls,
        ("LDC", "LDC", ("R2", "c[0x0][0x380]")),
        ("LOP3", "LOP3.LUT", ("R3", "R2", "0x7f", "RZ", "0xc0")),  # the low bits of a pointer
        ("LOP3", "LOP3.LUT", ("R4", "R2", "R0", "RZ", "0xfc")),  # a field or'ed over a pointer
    )
    assert ls.get("R3", {}) == ((), (0,) * 32)
    r4 = ls.get("R4", {})
    assert r4 is not None and r4[0] == ls.get("R2", {})[0] and r4[1] == tuple(range(32))


def test_a_compare_gives_both_destinations_per_lane() -> None:
    ls = lanes()
    tid(ls)
    run(ls, ("ISETP", "ISETP.GE.U32.AND", ("P0", "P1", "R0", "0x10", "PT")))
    assert ls.lanes_of("P0") == tuple(i >= 16 for i in range(32))
    assert ls.lanes_of("P1") == tuple(i < 16 for i in range(32))


def test_gap_names_the_opcode_that_broke_the_chain() -> None:
    ls = lanes()
    tid(ls)
    run(
        ls,
        ("LDG", "LDG.E", ("R5", "desc[UR0][R0.64]")),
        ("IADD", "IADD", ("R6", "R5", "0x10")),
    )
    addr = ls.address("[R6.64]", {})
    assert addr is None and ls.address_gap("[R6.64]", addr) == ("unknown", "LDG.E")


def test_a_spilled_value_comes_back_through_the_frame() -> None:
    ls = lanes()
    ls.spill = True
    tid(ls)
    run(
        ls,
        ("STL", "STL", ("[R1+0x8]", "R0")),
        ("LDG", "LDG.E", ("R0", "desc[UR6][R4.64]")),
        ("LDL", "LDL", ("R0", "[R1+0x8]")),
    )
    assert ls.get("R0", {}) == ((), tuple(range(32)))


def test_without_the_frame_a_reload_is_unknown() -> None:
    ls = lanes()
    tid(ls)
    run(ls, ("STL", "STL", ("[R1+0x8]", "R0")), ("LDL", "LDL", ("R0", "[R1+0x8]")))
    assert ls.get("R0", {}) is None


def test_a_slot_indexed_by_another_register_is_not_a_spill() -> None:
    ls = lanes()
    ls.spill = True
    tid(ls)
    run(ls, ("STL", "STL", ("[R5+0x8]", "R0")), ("LDL", "LDL", ("R0", "[R5+0x8]")))
    assert ls.get("R0", {}) is None


def _sectors_of(batch: str, case: str) -> tuple[float, float]:
    """A kernel's global load sectors per block: one middle block replayed, the counter's value."""
    from kta.models.gb10sim import kernel, probeset
    from kta.models.gb10sim.machine import default_machine
    from kta_perfmodel import addresses, table

    d = table.BATCHES / batch
    if not (d / "build").is_dir():
        pytest.skip("batch builds are local, not in Git (D914)")
    m = default_machine().with_overrides({"sim.l2_replay": False})
    keep, kernel.SM = kernel.SM, addresses._one_block  # type: ignore[assignment,misc]
    try:
        cases = {c.id: c for c in probeset.all_cases(d)}
        sm = cases[f"{case} kernel"].build(machine=m)
        sm.run()
    finally:
        kernel.SM = keep
    sm = getattr(sm, "sm", sm)
    c = json.loads((d / "measured.json").read_text())["counters"][case]
    return sm.count["ldg_sectors"], c[addresses.LD] / c["launch__grid_size"]


def test_loads_follow_thread_x_guards_and_lane_exits() -> None:
    """D903: threadIdx.x wraps at the block's first dimension, a guard per lane reads those lanes,
    an exit per lane ends those lanes: the sectors are the counter's to the sector."""
    for batch, case in (
        ("r18", "gemv18_np8_warm"),  # 2976 sectors: 2252 with lane 0 of a warp at tid 224
        ("r18", "gemv18_mk16_warm"),  # 1580: 1370
        ("r17", "gr_w8_p1280"),  # 20: the compiler sinks the load under `kr == 0`
    ):
        model, counter = _sectors_of(batch, case)
        assert model == counter, case

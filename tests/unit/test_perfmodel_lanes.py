"""The lanes as a class of integer instructions (round 15): divisions, pairs, bitwise."""

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

"""PERFMODEL: the GB10 machine description (machine.yaml) and the engine built from it."""

from __future__ import annotations

import pytest

from kta.core.errors import ConfigError
from kta.models.gb10sim.machine import default_machine
from kta.models.gb10sim.sass import parse
from kta.models.gb10sim.sm import SM

# the upstream opcode sets (agentic/gb10/sim/sm.py at 0eef497) the description replaced
FMA_FP32 = {"FFMA", "FADD", "FMUL"}
FMA_2 = {"IMAD", "HFMA2", "HADD2", "HMUL2", "IMUL"}
ALU_2 = {"LOP3", "PRMT", "SHF", "IADD3", "ISETP", "SEL", "LEA", "F2FP", "IMNMX", "FMNMX"}
ALU_2 |= {"FSETP", "FSEL", "PLOP3", "LOP", "IABS", "SHL", "SHR"}
ALU_2 |= {"I2FP"}  # upstream ee6534c (cvt.cu): int -> fp32 on the ALU
XU = {"F2I", "I2F", "F2F", "FRND", "F2IP", "MUFU"}


def test_opcode_classes_match_the_upstream_sets() -> None:
    m = default_machine()
    for op in FMA_FP32:
        c = m.classify(op, op)
        assert c.gates == ("fma",) and dict(c.hold) == {"fma": "sim.fma_fp32_clk"}
    for op in FMA_2:
        c = m.classify(op, op)
        assert c.gates == ("fma",) and dict(c.hold) == {"fma": "sim.alu_half_rate_clk"}
    for op in ALU_2:
        assert m.classify(op, op).gates == ("alu",)
    for op in XU:
        assert m.classify(op, op).gates == ("xu",)
    assert m.classify("HADD2", "HADD2.F32").either == ("alu", "fma")
    assert m.classify("RED", "RED.E.ADD.F32").name == "red"
    assert m.classify("RED", "RED.ADD").name == "red_other"
    assert m.classify("MOV", "MOV").name == "other"
    assert m.classify("HMMA", "HMMA.16816.F32").gates == ("tensor",)


def test_resources_and_scopes() -> None:
    m = default_machine()
    assert set(m.gating("smsp")) == {"tensor", "fma", "alu", "xu", "mio"}
    assert m.smsp_per_sm == 4 and m.sms == 48 and m.sms_per_tpc == 2
    assert "store_port" in m.resources("tpc")


def test_overrides_copy_and_reject_unknown_keys() -> None:
    m = default_machine()
    m2 = m.with_overrides({"sim.ldgsts_l2_latency_clk": 400, "clock_GHz": 2.4})
    assert m2.P["ldgsts_l2_latency_clk"] == 400 and m2.C["clock_GHz"] == 2.4
    assert m.P["ldgsts_l2_latency_clk"] == 350
    assert m2.status("sim.ldgsts_l2_latency_clk") == "override"
    assert m.status("sim.default_var_latency_clk") == "assumed"
    with pytest.raises(ConfigError):
        m.with_overrides({"sim.no_such_parameter": 1})


LOOP = """
\t.section\t.text.k,"ax",@progbits
        /*0000*/                   LDCU UR5, c[0x0][0x380] ;              /* 0x00007000ff0577ac */
                                                                          /* 0x000fe20008000800 */
        /*0010*/                   MOV R2, RZ ;                           /* 0x000000ff00027202 */
                                                                          /* 0x000fe20000000f00 */
.L_x_0:
        /*0020*/                   FFMA R4, R4, R4, R4 ;                  /* 0x0000000404047223 */
                                                                          /* 0x000fe20000000004 */
        /*0030*/                   IADD3 R2, PT, PT, R2, 0x1, RZ ;        /* 0x0000000102027810 */
                                                                          /* 0x000fe20007ffe0ff */
        /*0040*/                   ISETP.GE.AND P0, PT, R2, UR5, PT ;     /* 0x0000000502007c0c */
                                                                          /* 0x000fe2000bf06270 */
        /*0050*/              @!P0 BRA `(.L_x_0) ;                        /* 0xfffffffc00f08947 */
                                                                          /* 0x000fea000383ffff */
        /*0060*/                   EXIT ;                                 /* 0x000000000000794d */
                                                                          /* 0x000fea0003800000 */
"""


def test_kernel_arguments_decide_a_run_time_loop() -> None:
    ins = parse(LOOP, kernel="k")
    runs = {}
    for n in (5, 50):
        sm = SM(ins, 32, 1, 1, const_bank={(0, 0x380): n})
        runs[n] = sm.run()
    # the loop count comes from the declared argument, not from the fallback trip count
    assert runs[50]["issued"] - runs[5]["issued"] == 45 * 4


def test_trace_keeps_the_result_and_the_path_covers_every_cycle() -> None:
    from kta.models.gb10sim import attribution

    ins = parse(LOOP, kernel="k")
    plain = SM(ins, 128, 2, 1, const_bank={(0, 0x380): 20}).run()
    sm = SM(ins, 128, 2, 1, const_bank={(0, 0x380): 20}, resident=1, trace=True)
    traced = sm.run()
    sm2 = SM(ins, 128, 2, 1, const_bank={(0, 0x380): 20}, resident=1)
    assert traced["cycles"] == sm2.run()["cycles"] and plain["issued"] == traced["issued"]
    a = attribution.analyse(sm, sm.cycles, {"launch": (100.0, "launch_stream_us")})
    assert a["covered_cycles"] == sm.cycles
    assert abs(sum(d["share"] for d in a["path"]) - 1.0) < 1e-9
    # the second block starts when the first exits: the path crosses the block replacement
    assert any(d["resource"] == "cta_dispatch" for d in a["path"])
    assert a["ledger"]["assumed_or_fitted_on_path"]["launch_stream_us"]["status"] == "fitted"

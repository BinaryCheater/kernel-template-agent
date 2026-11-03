"""WP26: the copied GB10 simulator's parameter file and SASS reader (D680-D683)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kta.core.errors import NotSupported
from kta.models.gb10sim import model
from kta.models.gb10sim.l3 import simulate_build
from kta.models.gb10sim.sass import main_loop, parse

STATUSES = {"measured", "fitted", "structural", "assumed", "withdrawn", "unidentified"}  # LOOP §3

# Two instructions of a loop with its back edge, as cuobjdump -sass prints them (numeric
# target, "Function :" header) and as nvdisasm -hex prints them (label target, .text section).
CUOBJDUMP = """
\tFunction : other_kernel
        /*0000*/                   NOP ;                                  /* 0x0000000000007918 */
                                                                          /* 0x000fc00000000000 */
\tFunction : gemm_kernel
        /*0000*/                   HMMA.16816.F32 R4, R8, R12, R4 ;       /* 0x0000000c0804723c */
                                                                          /* 0x004ff00000001804 */
        /*0010*/               @P0 BRA 0x0 ;                              /* 0xfffffffc00c08947 */
                                                                          /* 0x000fea000383ffff */
"""
NVDISASM = """
\t.section\t.text.gemm_kernel,"ax",@progbits
.L_x_0:
        /*0000*/                   HMMA.16816.F32 R4, R8, R12, R4 ;       /* 0x0000000c0804723c */
                                                                          /* 0x004ff00000001804 */
        /*0010*/               @P0 BRA `(.L_x_0) ;                        /* 0xfffffffc00c08947 */
                                                                          /* 0x000fea000383ffff */
"""


def test_every_parameter_has_a_status() -> None:
    raw = json.loads(model.PARAMS.read_text())
    for key, entry in raw.items():
        if key.startswith("_"):
            continue
        assert entry["status"] in STATUSES, key
    sim = raw["sim"]
    assert set(sim["value_status"]) == set(sim["value"])
    assert set(sim["value_status"].values()) <= STATUSES


def test_code_constants_are_parameters() -> None:
    for key in (
        "clock_under_load_GHz",
        "launch_stream_us",
        "graph_launch_us",
        "dram_datasheet_GBs",
    ):
        assert key in model.C


@pytest.mark.parametrize("text", [CUOBJDUMP, NVDISASM], ids=["cuobjdump", "nvdisasm"])
def test_reader_finds_the_function_and_its_loop(text: str) -> None:
    ins = parse(text, kernel="gemm_kernel")
    assert [x.base for x in ins] == ["HMMA", "BRA"]
    assert ins[1].target == 0
    assert main_loop(ins) == (0, 1)


def test_l3_refuses_a_lowering_it_does_not_map(tmp_path: Path) -> None:
    sass = tmp_path / "k.sass"
    sass.write_text(CUOBJDUMP)
    holes = {"block_M": 128, "block_N": 128, "block_K": 32, "num_stages": 3, "threads": 128}
    with pytest.raises(NotSupported, match=r"cp\.async"):
        simulate_build(sass, "gemm_kernel", holes, {"M": 128, "N": 128, "K": 32}, 64, 0, True)
    with pytest.raises(NotSupported, match="holes"):
        simulate_build(sass, "gemm_kernel", {}, {"M": 128, "N": 128, "K": 32}, 64, 0, True)

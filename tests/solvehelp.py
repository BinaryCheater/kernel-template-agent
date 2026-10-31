"""Builders for formulation/solve/model tests: the GEMM test family over the seed
Templates, a small linear family, and test HwModels.

``TEST_HW`` is a schema-1 HwModel (peaks only, prices legacy cost terms). ``TEST_ARCH`` is
the seed architecture class (knowledge/seed/hardware) and ``TEST_HW2`` a chip instance on
it whose numbers are test values, not GB10 facts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from knowhelp import seed, template
from kta.core.config import repo_root
from kta.core.context import Context
from kta.core.schemas.evaluation import Case
from kta.core.schemas.hardware import HwModel
from kta.core.schemas.hwmodel import HwArch
from kta.core.schemas.knowledge import Chain, Family, Template
from kta.formulation import Formulation, assemble
from kta.knowledge import FamilyPackage, load_family
from kta.knowledge.pool import find_template
from kta.models import MachineModel
from kta.store.base import Store

GEMM_DIR = repo_root() / "tests" / "acceptance" / "WP5" / "gemm_family"

TEST_HW = HwModel(
    target="test-gpu",
    version=1,
    peaks={"dram": 2.5e11, "fma": 1.0e13, "issue": 4.0e11, "smem": 3.0e10, "tensor": 2.0e14},
    limits={
        "smem_per_cta": 101376,
        "smem_per_sm": 102400,
        "smem_reserved_per_cta": 1024,
        "smem_queue_depth": 1,
        "smem_banks": 32,
        "smem_bank_bytes": 4,
        "max_ctas_per_sm": 32,
        "max_threads_per_sm": 1536,
        "regs_per_sm": 65536,
    },
)


TEST_ARCH = HwArch.model_validate(
    yaml.safe_load(
        (repo_root() / "knowledge" / "seed" / "hardware" / "arch_sm12x.yaml").read_text("utf-8")
    )
)
_PARAMS: dict[str, int | float] = {
    "sm_count": 4,
    "smsp_per_sm": 4,
    "sm_clock_hz": 1.0e9,
    "dram_bytes_per_s": 2.5e11,
    "l2_bytes_per_s": 1.0e12,
    "smem_bytes_per_s_per_sm": 1.28e11,
    "smem_wavefront_bytes": 128,
    "fp32_flops_per_s_per_sm": 5.0e12,
    "tensor_fp16_flops_per_s": 2.0e14,
    "tensor_bf16_flops_per_s": 2.0e14,
    "tensor_tf32_flops_per_s": 1.0e14,
    "sfu_ops_per_cycle_per_sm": 16,
    "kernel_launch_s": 2.0e-6,
    "dram_latency_ns": 300.0,
    "l2_latency_ns": 150.0,
    "async_copy_latency_s": 2.5e-7,
    "tma_latency_s": 5.0e-7,
    "mma_latency_cycles": 32,
    "barrier_latency_cycles": 40,
}
_LIMITS = {**TEST_HW.limits, **_PARAMS}
TEST_HW2 = HwModel(
    target="test-gpu",
    version=2,
    arch=TEST_ARCH.ref,
    limits=_LIMITS,
    provenance={f"limits.{k}": {"status": "default", "source": "test value"} for k in _LIMITS},  # type: ignore[arg-type]  # validated into Provenance
)
TEST_MM = MachineModel.build(TEST_HW2, TEST_ARCH)


def case(*shapes: dict[str, int]) -> Case:
    return Case(
        problem="gemm_fp16_test",
        shape_dist=[{"shape": s, "weight": 1.0} for s in shapes],
        objective="weighted_latency",
        target="test-gpu",
    )


def chain_templates(store: Store, chain: Chain) -> dict[str, Template]:
    out: dict[str, Template] = {}
    for item in chain.items:
        hit = find_template(store, item.template)
        assert hit is not None, item.template
        out[item.template] = hit[1]
    return out


def gemm(ctx: Context, store: Store) -> tuple[FamilyPackage, dict[str, Template], Formulation]:
    seed(ctx, store)
    fp = load_family(GEMM_DIR)
    templates = chain_templates(store, fp.chain)
    f = assemble(fp, fp.chain, templates, case({"M": 512, "N": 512, "K": 512}), TEST_HW.limits)
    return fp, templates, f


def _var(name: str, lo: int, hi: int) -> dict[str, Any]:
    return {
        "name": name,
        "kind": "numeric",
        "binding": "compile",
        "role": "software",
        "domain": {"lo": lo, "hi": hi},
        "meaning": name,
    }


LINEAR_TEMPLATE = template(
    "reduction_partition",
    name="linear_test",
    variables=[_var("u", 0, 1), _var("v", 0, 1)],
    constraints=[
        {"id": "c_budget", "expr": "u + 2 * v <= p.cap", "meaning": "budget", "cls": "resource"},
        {"id": "c_mix", "expr": "3 * u - v >= 10", "meaning": "mix", "cls": "semantic"},
    ],
    preconditions=[],
    cost=[
        {"unit": "dram", "work": "p.base - 997 * u + 13 * v", "overlap": "overlappable"},
        {"unit": "fma", "work": "2003 * u + 4001 * v + 1", "overlap": "overlappable"},
        {"unit": "issue", "work": "3 * u + 5 * v", "overlap": "serial"},
    ],
)


def linear_family(hi: int) -> tuple[FamilyPackage, dict[str, Template]]:
    """Two range variables x, y in [1, hi] bound to one linear test Template.

    Shapes give ``cap`` (budget) and ``base`` (DRAM work at x = 0; keep it > 997 * hi).
    """
    family = Family.model_validate(
        {
            "name": "linear_test",
            "version": 1,
            "backend": "cuda",
            "variables": [_var("x", 1, hi), _var("y", 1, hi)],
            "entry": "gen.py:generate",
        }
    )
    chain = Chain.model_validate(
        {
            "workload": "linear test",
            "items": [
                {
                    "key": "lin",
                    "template": "linear_test@1",
                    "applies_to": "test",
                    "bindings": {"u": "x", "v": "y", "p.cap": "case.cap", "p.base": "case.base"},
                }
            ],
        }
    )
    fp = FamilyPackage(Path("."), family, chain, [], "", Path("gen.py"), "generate")
    return fp, {"linear_test@1": LINEAR_TEMPLATE}

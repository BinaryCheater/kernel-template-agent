"""Toy knowledge for the WP9a fixture store (D94: very simple cases are fine).

# kta:fixture synthetic records for frontend development before WP7's real runs (D95, D168).
Every record written here has origin "fixture"; none of it is evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kta.core.schemas import Actor, Record
from kta.store import SqliteStore

SYSTEM = Actor(type="system", name="kta-devseed")
CHECKER = Actor(type="checker", name="validate_template@1")


@dataclass
class Seeder:
    store: SqliteStore
    clock: Any
    ids: dict[str, str] = field(default_factory=lambda: dict[str, str]())

    def put(
        self,
        key: str,
        kind: str,
        body: dict[str, Any],
        refs: dict[str, Any] | None = None,
        actor: Actor = SYSTEM,
    ) -> str:
        rec = Record.new(kind, body, actor, self.clock.now(), refs=refs or {}, origin="fixture")
        rid = self.store.put(rec)
        self.ids[key] = rid
        return rid

    def status(
        self,
        subject: str,
        actor: Actor,
        to_status: str | None = None,
        to_assurance: str | None = None,
    ) -> None:
        body: dict[str, Any] = {"subject": subject, "reason": "fixture: seeded verdict"}
        if to_status:
            body["to_status"] = to_status
        if to_assurance:
            body["to_assurance"] = to_assurance
        self.store.put(Record.new("StatusChange", body, actor, self.clock.now(), origin="fixture"))


def var(name: str, domain: list[Any], meaning: str, binding: str = "compile") -> dict[str, Any]:
    return {
        "name": name,
        "kind": "structure",
        "binding": binding,
        "role": "software",
        "domain": domain,
        "meaning": meaning,
    }


# name, version, level_from, level_to, change, refines, assurance, summary, requires_hw, variables
TEMPLATES: list[tuple[str, int, int, int, str, str, str, str, list[str], list[dict[str, Any]]]] = [
    (
        "grid_stride_map",
        1,
        1,
        2,
        "vertical",
        "map",
        "tested",
        "Map each element to a thread with a grid-stride loop so any n fits a fixed grid.",
        [],
        [
            var("block", [128, 256, 512], "threads per CTA"),
            var("unroll", [1, 2, 4], "iterations per thread"),
        ],
    ),
    (
        "vector_load",
        2,
        2,
        3,
        "vertical",
        "transfer",
        "tested",
        "Load contiguous elements with 64- or 128-bit global loads when alignment allows.",
        ["ldg_128"],
        [var("vec", [1, 2, 4, 8], "elements per load")],
    ),
    (
        "coalesced_store",
        1,
        2,
        3,
        "vertical",
        "store",
        "tested",
        "Write results so consecutive threads write consecutive addresses.",
        [],
        [var("vec", [1, 2, 4], "elements per store")],
    ),
    (
        "row_per_cta",
        1,
        1,
        2,
        "vertical",
        "map",
        "tested",
        "Assign one row of a row-wise operator to one CTA.",
        [],
        [var("threads", [128, 256, 512, 1024], "threads per CTA")],
    ),
    (
        "online_softmax",
        1,
        1,
        1,
        "horizontal",
        "reduce",
        "bounded",
        "Fuse max and sum into one pass using the rescalable exponential sum.",
        [],
        [],
    ),
    (
        "warp_reduce",
        1,
        2,
        3,
        "vertical",
        "reduce",
        "tested",
        "Reduce within a warp with shuffle instructions.",
        ["shfl_sync"],
        [],
    ),
    (
        "block_reduce_smem",
        1,
        2,
        3,
        "vertical",
        "reduce",
        "tested",
        "Combine per-warp partials through shared memory.",
        [],
        [],
    ),
    (
        "cta_tile",
        1,
        1,
        2,
        "vertical",
        "contract",
        "tested",
        "Tile the output into BM x BN blocks per CTA and loop over K in BK steps.",
        [],
        [
            var("BM", [32, 64, 128], "CTA tile rows"),
            var("BN", [32, 64, 128], "CTA tile cols"),
            var("BK", [16, 32, 64], "K step"),
        ],
    ),
    (
        "warp_tile",
        1,
        2,
        3,
        "vertical",
        "contract",
        "tested",
        "Split the CTA tile among warps; each warp owns a WM x WN sub-tile.",
        [],
        [var("WM", [16, 32, 64], "warp tile rows"), var("WN", [16, 32, 64], "warp tile cols")],
    ),
    (
        "mma_bf16",
        1,
        3,
        5,
        "vertical",
        "contract",
        "tested",
        "Issue mma.sync m16n8k16 bf16 with fp32 accumulation on the warp tile.",
        ["mma_sync_bf16"],
        [],
    ),
    (
        "epilogue_store",
        1,
        2,
        3,
        "vertical",
        "store",
        "tested",
        "Convert accumulators and store the C tile with vectorized writes.",
        [],
        [var("vec", [2, 4, 8], "elements per store")],
    ),
    (
        "smem_double_buffer",
        1,
        3,
        3,
        "expansion",
        "transfer",
        "declared",
        "Overlap the next K tile's cp.async copy with the current tile's math.",
        ["cp_async"],
        [var("stages", [2, 3], "pipeline stages")],
    ),
]

# family, version, backend, problem, workload,
# items (key, template, applies_to, after), variables, glue
FAMILIES: list[
    tuple[
        str,
        int,
        str,
        str,
        str,
        list[tuple[str, str, str, list[str]]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]
] = [
    (
        "vadd_f32",
        1,
        "cuda",
        "vadd_f32",
        "c[i] = a[i] + b[i], fp32",
        [
            ("grid", "grid_stride_map@1", "elementwise add over n", []),
            ("ld", "vector_load@2", "loads of a and b", ["grid"]),
            ("st", "coalesced_store@1", "store of c", ["grid"]),
        ],
        [
            var("block", [128, 256, 512], "threads per CTA"),
            var("unroll", [1, 2, 4], "iterations per thread"),
            var("vec", [1, 2, 4], "elements per access"),
        ],
        [
            {
                "id": "g1",
                "expr": "vec <= 4",
                "meaning": "Load and store widths are tied together, so no single step owns this",
                "cls": "implementation",
            }
        ],
    ),
    (
        "softmax_rowwise",
        2,
        "tilelang",
        "softmax_bf16",
        "y = softmax(x, dim=-1), bf16 rows",
        [
            ("row", "row_per_cta@1", "one row per CTA", []),
            ("online", "online_softmax@1", "max+sum in one pass", ["row"]),
            ("ld", "vector_load@2", "row loads", ["row"]),
            ("wred", "warp_reduce@1", "per-warp max and sum", ["online"]),
            ("bred", "block_reduce_smem@1", "cross-warp combine", ["wred"]),
            ("st", "coalesced_store@1", "normalized row store", ["bred", "ld"]),
        ],
        [var("threads", [128, 256, 512, 1024], "threads per CTA")],
        [],
    ),
    (
        "gemm_tiled",
        1,
        "tilelang",
        "gemm_bf16",
        "C = A @ B, bf16 in, fp32 accumulate",
        [
            ("cta", "cta_tile@1", "C tiling over (M,N), K loop", []),
            ("warp", "warp_tile@1", "warp sub-tiles", ["cta"]),
            ("a_ld", "vector_load@2", "A tile global to smem", ["cta"]),
            ("b_ld", "vector_load@2", "B tile global to smem", ["cta"]),
            ("mma", "mma_bf16@1", "warp-tile multiply-accumulate", ["warp", "a_ld", "b_ld"]),
            ("epi", "epilogue_store@1", "C tile store", ["mma"]),
        ],
        [
            var("BM", [32, 64, 128], "CTA tile rows"),
            var("BN", [32, 64, 128], "CTA tile cols"),
            var("BK", [16, 32, 64], "K step"),
        ],
        [
            {
                "id": "g1",
                "expr": "(BM*BK + BK*BN)*2 <= hw.smem_per_cta",
                "meaning": "Shared memory is shared by the A and B load steps",
                "cls": "resource",
            }
        ],
    ),
]

PROBLEMS = ["vadd_f32", "softmax_bf16", "gemm_bf16", "rmsnorm_bf16"]
# case key, problem, shape_dist
CASES: list[tuple[str, str, list[dict[str, Any]]]] = [
    ("vadd", "vadd_f32", [{"shape": {"n": 16777216}, "weight": 1.0}]),
    ("softmax", "softmax_bf16", [{"shape": {"rows": 4096, "cols": 1024}, "weight": 1.0}]),
    ("gemm", "gemm_bf16", [{"shape": {"M": 256, "N": 256, "K": 256}, "weight": 1.0}]),
    ("rmsnorm", "rmsnorm_bf16", [{"shape": {"rows": 2048, "cols": 4096}, "weight": 1.0}]),
]
BASELINES = ["torch eager", "cub::DeviceTransform", "triton tutorial", "torch.matmul (cuBLAS)"]
REFERENCES = ["triton_softmax_tutorial", "cutlass_sgemm_example", "cub_block_reduce"]


def seed_knowledge(s: Seeder) -> None:
    src = s.put(
        "source:seed",
        "Source",
        {
            "name": "toy seed for WP9a",
            "type": "other",
            "locator": "tools/kta_devseed",
            "revision": "fixture",
        },
    )
    for name in BASELINES:
        s.put(
            f"baseline:{name}",
            "Source",
            {"name": name, "type": "other", "locator": "fixture", "revision": "fixture"},
        )
    for p in PROBLEMS:
        s.put(f"problem:{p}", "Problem", {"name": p, "reference_impl": f"{p}_ref.py"})
    for key, problem, dist in CASES:
        s.put(
            f"case:{key}",
            "Case",
            {
                "problem": problem,
                "shape_dist": dist,
                "objective": "weighted_latency",
                "target": "gb10",
            },
            refs={"problem": s.ids[f"problem:{problem}"]},
        )
    for name, ver, lf, lt, change, refines, assurance, summary, hw, variables in TEMPLATES:
        tid = s.put(
            f"template:{name}@{ver}",
            "Template",
            {
                "name": name,
                "version": ver,
                "level_from": lf,
                "level_to": lt,
                "change": change,
                "refines": refines,
                "object": {"type": refines, "description": f"the {refines} transaction"},
                "variables": variables,
                "program": {"description": summary},
                "backward_map": "identity on the refined transaction",
                "scope": {"predicate": "True", "text": "toy scope"},
                "requires_hw": hw,
                "signature": {"algorithm": name},
                "summary": summary,
            },
            refs={"sources": [src]},
        )
        if assurance != "declared":
            s.status(tid, CHECKER, to_status="reproduced", to_assurance=assurance)
    for fam, ver, backend, problem, workload, items, variables, glue in FAMILIES:
        tpl = sorted({s.ids[f"template:{t}"] for _, t, _, _ in items})
        chain = s.put(
            f"chain:{fam}@{ver}",
            "Chain",
            {
                "workload": workload,
                "items": [
                    {"key": k, "template": t, "applies_to": a, "after": after}
                    for k, t, a, after in items
                ],
            },
            refs={"templates": tpl},
        )
        s.put(
            f"family:{fam}@{ver}",
            "Family",
            {
                "name": fam,
                "version": ver,
                "backend": backend,
                "variables": variables,
                "glue_constraints": glue,
                "entry": "gen.py:generate",
            },
            refs={"chain": chain, "problem": s.ids[f"problem:{problem}"]},
        )
    for ref in REFERENCES:
        s.put(
            f"reference:{ref}",
            "Reference",
            {"name": ref, "problem": "fixture", "build": "fixture", "runnable_on": ["gb10"]},
            refs={"source": src},
        )

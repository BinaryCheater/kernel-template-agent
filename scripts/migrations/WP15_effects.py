# ruff: noqa: E501  (the effects table is YAML text kept one line per entry)
"""WP15 seed migration: Template ``cost`` -> ``effects`` (decisions D460, D468).

Replaces the top-level ``cost:`` block of every seed Template YAML under
``knowledge/seed`` with the hand-written ``effects:`` block below, keeping the rest of the
file (comments included) byte for byte. Idempotent: a file whose ``effects:`` block
already equals the one below is left alone; a file with neither block, or with a
template this table does not know, is an error.

Stored records are never touched (they upcast on read, D461). Versions are kept: the
seed is migrated in place (D468), so a store that imported the pre-WP15 seed reports a
conflict on re-import; no such store holds real data (D320, D325).

Run from the repository root: ``uv run python scripts/migrations/WP15_effects.py``
(``--check`` exits 1 when a file would change).
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = ROOT / "knowledge" / "seed"
BLOCK = re.compile(r"^(cost|effects):.*?(?=^[a-z_]+:)", re.S | re.M)

# Shared fragments.
SERIAL = '    - {{between: [{0}], when: "False", note: "never overlaps (v1 cost was serial)"}}\n'


def _effects(txs: str, storage: str = "", overlap: str = "") -> str:
    out = "effects:\n"
    out += "  transactions:\n" + txs if txs else "  transactions: []\n"
    out += "  storage:\n" + storage if storage else "  storage: []\n"
    out += "  overlap:\n" + overlap if overlap else "  overlap: []\n"
    return out


EFFECTS: dict[str, str] = {
    # ---- the paper's GEMM steps and the two contraction-cost Templates (D280) ---------
    "templates/async_transfer.yaml": _effects(
        "    # One copy operation per granularity_bytes; which operation class depends on\n"
        "    # the mechanism. The shared-memory side is priced by shared_resource_binding.\n"
        "    - {id: copy, pattern: transfer, kind: mechanism,\n"
        "       kind_map: {software: global_load, instruction: cp_async_16, controller: tma_bulk},\n"
        '       count: "p.size_bytes / granularity_bytes",\n'
        '       amounts: {bytes: "granularity_bytes", vector_bytes: "granularity_bytes"},\n'
        "       src: global, dst: smem}\n"
    ),
    "templates/compute_instruction_selection.yaml": _effects(
        "    # p.M x p.N x p.K is one executor tile's contraction; p.instances counts tiles.\n"
        "    # The contraction is declared here once (it was missing from WP5's chain).\n"
        "    - {id: mma, pattern: contract, kind: instruction,\n"
        '       count: "p.M * p.N * p.K / macs_per_instruction",\n'
        '       amounts: {flops: "2 * macs_per_instruction"}}\n'
    ),
    "templates/multi_buffer_rotation.yaml": _effects(
        '    - {id: barrier, pattern: sync, kind: cta_barrier, count: "p.iterations * (1 + beta)",\n'
        '       amounts: {warps: "p.warps"}, note: "leading barrier, plus the trailing one when beta"}\n',
        overlap=(
            "    # Loads of generation p+P overlap the consumption of generation p only when\n"
            "    # P >= 1 (the v1 cost forced this into a DRAM term); with P in flight the\n"
            "    # loop is also bound by the load-to-use recurrence.\n"
            '    - {between: ["pattern:transfer", "pattern:contract"], when: "P >= 1",\n'
            '       inflight: "P", trips: "p.iterations"}\n'
        ),
    ),
    "templates/output_tiling.yaml": _effects(
        "    - {id: ctas, pattern: launch, kind: cta_launch,\n"
        '       count: "ceil_div(p.M, tile_m) * ceil_div(p.N, tile_n)"}\n'
        '    - {id: kernel, pattern: launch, kind: kernel_launch, count: "1"}\n'
        "    # Lower estimate of DRAM traffic: every input and output byte once. The tiles'\n"
        "    # operand reads from L2 are declared by the transfer step that performs them.\n"
        '    - {id: compulsory, pattern: transfer, kind: dram_stream, count: "1",\n'
        '       amounts: {bytes: "p.compulsory_bytes"}, src: dram, dst: l2}\n'
    ),
    "templates/physical_storage.yaml": _effects(
        "",
        storage='    - {id: buffer, level: level, scope: cta, bytes: "p.size_bytes * p.copies"}\n',
    ),
    "templates/reduction_partition.yaml": _effects(
        "    # Combining the groups' partial results; a sequential sum into one accumulator\n"
        "    # needs no combine step.\n"
        '    - {id: combine, pattern: reduce, kind: ffma, count: "(num_groups - 1) * p.outputs",\n'
        '       amounts: {ops: "1"}, when: "assignment != \'contiguous_sequential\'"}\n'
    ),
    "templates/shared_resource_binding.yaml": _effects(
        '    - {id: access, pattern: access, kind: smem_access, count: "p.accesses",\n'
        '       amounts: {wavefronts: "p.wavefronts"}}\n'
    ),
    # ---- WP10a reference Templates (analysis only) --------------------------------------
    "references/cutlass_ampere_tf32_gemm/templates/block_rasterization.yaml": _effects(
        '    - {id: operand_reads, pattern: transfer, kind: l2_traffic, count: "1",\n'
        '       amounts: {bytes: "p.operand_bytes * p.tiles / group"}}\n'
    ),
    "references/cutlass_ampere_tf32_gemm/templates/boundary_predication.yaml": _effects(
        '    - {id: predicates, pattern: other, kind: warp_instruction, count: "p.tile_accesses"}\n'
    ),
    "references/cutlass_ampere_tf32_gemm/templates/collective_fragment_load.yaml": _effects(
        '    - {id: ldmatrix, pattern: transfer, kind: smem_access, count: "p.tile_bytes / 512",\n'
        '       amounts: {wavefronts: "4"}, note: "ldmatrix.x4: 512 B per warp, 4 wavefronts"}\n'
    ),
    "references/cutlass_ampere_tf32_gemm/templates/compute_instruction_selection.yaml": _effects(
        "    - {id: mma, pattern: contract, kind: instruction,\n"
        '       count: "p.M * p.N * p.K / macs_per_instruction",\n'
        '       amounts: {flops: "2 * macs_per_instruction"}}\n'
    ),
    "references/cutlass_ampere_tf32_gemm/templates/operand_precision_conversion.yaml": _effects(
        '    - {id: convert, pattern: map, kind: ffma, count: "p.M * p.K + p.K * p.N",\n'
        '       amounts: {ops: "1"}}\n'
    ),
    "references/cutlass_ampere_tf32_gemm/templates/operator_fusion.yaml": _effects(
        '    - {id: traffic, pattern: transfer, kind: dram_stream, count: "1",\n'
        '       amounts: {bytes: "p.producer_input_bytes + p.consumer_extra_input_bytes'
        ' + p.output_bytes"}}\n'
    ),
    "references/cutlass_ampere_tf32_gemm/templates/output_tiling.yaml": _effects(
        "    - {id: ctas, pattern: launch, kind: cta_launch,\n"
        '       count: "ceil_div(p.M, tile_m) * ceil_div(p.N, tile_n)"}\n'
    ),
    "references/cutlass_ampere_tf32_gemm/templates/staged_relayout.yaml": _effects(
        '    - {id: relayout, pattern: transfer, kind: smem_access, count: "2 * p.tile_bytes / 512",\n'
        '       amounts: {wavefronts: "4"}}\n',
        overlap=SERIAL.format("relayout"),
    ),
    "references/cutlass_ampere_tf32_gemm/templates/vectorized_access.yaml": _effects(
        '    - {id: load, pattern: transfer, kind: global_load, count: "1",\n'
        '       amounts: {bytes: "p.tile_bytes", vector_bytes: "vector_bytes"}}\n'
    ),
    "references/flash_attention_2_fwd/templates/accumulator_operand_reuse.yaml": _effects(
        "", overlap=""
    ).replace(
        "  transactions: []\n",
        "  # The step removes a shared-memory round trip; it adds no transaction.\n"
        "  transactions: []\n",
    ),
    "references/flash_attention_2_fwd/templates/approximate_math_rewrite.yaml": _effects(
        '    - {id: exp2, pattern: map, kind: mufu, count: "p.elements", amounts: {ops: "1"}}\n'
    ),
    "references/flash_attention_2_fwd/templates/cross_lane_reduction.yaml": _effects(
        '    - {id: shuffle, pattern: reduce, kind: warp_shuffle, count: "p.reductions * log2(lanes)"}\n'
    ),
    "references/flash_attention_2_fwd/templates/masked_tile_pruning.yaml": _effects(
        '    - {id: kept, pattern: contract, kind: tensor_flops, count: "1",\n'
        '       amounts: {flops: "p.work_full * p.mask_density"}}\n'
    ),
    "references/flash_attention_2_fwd/templates/online_rescaled_reduction.yaml": _effects(
        '    - {id: rescale, pattern: reduce, kind: ffma, count: "p.rows * p.tiles * p.head_dim",\n'
        '       amounts: {ops: "1"}}\n'
        '    - {id: exp, pattern: map, kind: mufu, count: "p.rows * p.tiles", amounts: {ops: "1"}}\n'
    ),
    "references/flash_attention_3_fwd/templates/cache_eviction_hint.yaml": _effects(
        '    - {id: reuse, pattern: transfer, kind: l2_traffic, count: "1",\n'
        '       amounts: {bytes: "p.reused_bytes"}}\n'
    ),
    "references/flash_attention_3_fwd/templates/cluster_multicast.yaml": _effects(
        '    - {id: multicast, pattern: transfer, kind: l2_traffic, count: "1",\n'
        '       amounts: {bytes: "p.shared_tile_bytes / cluster_size"}}\n'
    ),
    "references/flash_attention_3_fwd/templates/cross_iteration_software_pipelining.yaml": _effects(
        '    - {id: softmax, pattern: map, kind: mufu, count: "p.softmax_work", amounts: {ops: "1"}}\n'
    ),
    "references/flash_attention_3_fwd/templates/per_slot_handshake.yaml": _effects(
        '    - {id: handshake, pattern: sync, kind: mbarrier_sync, count: "4 * p.tiles"}\n'
    ),
    "references/flash_attention_3_fwd/templates/persistent_tile_scheduler.yaml": _effects(
        '    - {id: ctas, pattern: launch, kind: cta_launch, count: "grid"}\n'
    ),
    "references/flash_attention_3_fwd/templates/pingpong_schedule.yaml": _effects(""),
    "references/flash_attention_3_fwd/templates/register_rebalancing.yaml": _effects(""),
    "references/flash_attention_3_fwd/templates/warp_specialization.yaml": _effects(
        "    - {id: producer, pattern: transfer, kind: warp_instruction,\n"
        '       count: "p.tiles * p.loads_per_tile"}\n'
    ),
    "references/flashinfer_batch_decode/templates/indexed_gather.yaml": _effects(
        '    - {id: gather, pattern: transfer, kind: dram_stream, count: "1",\n'
        '       amounts: {bytes: "p.gathered_bytes + p.index_bytes"}}\n'
    ),
    "references/flashinfer_batch_decode/templates/runtime_work_descriptors.yaml": _effects(
        '    - {id: planner, pattern: launch, kind: kernel_launch, count: "p.planner_launches"}\n'
    ),
    "references/flashinfer_batch_decode/templates/split_combine_kernel.yaml": _effects(
        '    - {id: partials, pattern: transfer, kind: dram_stream, count: "1",\n'
        '       amounts: {bytes: "2 * p.partial_bytes"}}\n'
        '    - {id: combine_kernel, pattern: launch, kind: kernel_launch, count: "1"}\n',
        overlap=SERIAL.format("partials"),
    ),
    "references/marlin_int4_gemm/templates/fused_dequantization.yaml": _effects(
        '    - {id: dequant, pattern: map, kind: ffma, count: "p.K * p.N * p.m_tiles / 2",\n'
        '       amounts: {ops: "1"}}\n'
        '    - {id: weights, pattern: transfer, kind: dram_stream, count: "1",\n'
        '       amounts: {bytes: "p.K * p.N / 2"}}\n'
    ),
    "references/marlin_int4_gemm/templates/linearized_work_partition.yaml": _effects(
        '    - {id: ctas, pattern: launch, kind: cta_launch, count: "grid"}\n'
    ),
    "references/marlin_int4_gemm/templates/offline_operand_prepack.yaml": _effects(
        '    - {id: packed, pattern: transfer, kind: dram_stream, count: "1",\n'
        '       amounts: {bytes: "p.K * p.N * pack_bits / 8"}}\n'
    ),
    "references/marlin_int4_gemm/templates/ordered_global_combine.yaml": _effects(
        '    - {id: combine, pattern: reduce, kind: l2_traffic, count: "1",\n'
        '       amounts: {bytes: "2 * p.tile_bytes * p.partials_per_tile"}}\n',
        overlap=SERIAL.format("combine"),
    ),
    "references/marlin_int4_gemm/templates/smem_partial_combine.yaml": _effects(
        "    - {id: combine, pattern: reduce, kind: smem_access,\n"
        '       count: "2 * p.partial_bytes * participants / 512", amounts: {wavefronts: "4"}}\n',
        overlap=SERIAL.format("combine"),
    ),
}


def migrate(path: Path, effects: str) -> bool:
    """Rewrite one file; True when it changed."""
    text = path.read_text(encoding="utf-8")
    found = BLOCK.search(text)
    if found is None:
        raise SystemExit(f"{path}: no top-level cost: or effects: block")
    if found.group(0) == effects:
        return False
    path.write_text(text[: found.start()] + effects + text[found.end() :], encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report, do not write")
    args = parser.parse_args()
    files = sorted(
        str(p.relative_to(SEED))
        for p in [*SEED.glob("templates/*.yaml"), *SEED.glob("references/*/templates/*.yaml")]
    )
    unknown = [f for f in files if f not in EFFECTS]
    if unknown:
        raise SystemExit(f"no effects written for {unknown}")
    changed: list[str] = []
    for rel in files:
        path = SEED / rel
        if args.check:
            text = path.read_text(encoding="utf-8")
            found = BLOCK.search(text)
            if found is None or found.group(0) != EFFECTS[rel]:
                changed.append(rel)
        elif migrate(path, EFFECTS[rel]):
            changed.append(rel)
    verb = "would change" if args.check else "changed"
    sys.stdout.write(f"{verb} {len(changed)} of {len(files)} seed Templates\n")
    for rel in changed:
        sys.stdout.write(f"  {rel}\n")
    return 1 if args.check and changed else 0


if __name__ == "__main__":
    raise SystemExit(main())

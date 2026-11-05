"""Evaluator-only source-version reconciliation for the selected FA2 branch."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


def reconciliation(repo: Path) -> dict:
    retained = repo / "knowledge/studies/flash-attention/inputs/sources.json"
    source = json.loads(retained.read_text())
    tree = Path(source["tree"]) / "csrc/flash_attn/src"
    # Each row pins a concrete current source witness and historical decision IDs.
    definitions = [
        (
            "output_tiling@2",
            "T01,T02,T04",
            "flash_fwd_launch_template.h",
            60,
            64,
            "dim3 grid",
            "CTA owns a query-row tile; head ratio is one. "
            "kernel_traits.h:74-77 gives warp row ownership. Tile dimensions "
            "remain caller-independent choices.",
            "coverage_exact_once,index_map_bijective",
        ),
        (
            "reduction_partition@1",
            "T13",
            "flash_fwd_kernel.h",
            274,
            309,
            "int n_block = n_block_max - 1",
            "Key blocks form a contiguous partition and are visited "
            "in descending order. Source order is not supplied as an answer to "
            "implementation sessions.",
            "exact_once,combine_complete",
        ),
        (
            "compute_instruction_selection@2",
            "T05,T11",
            "kernel_traits.h",
            29,
            41,
            "SM80_16x8x16_F32F16F16F32_TN",
            "Current source uses FP16 MMA with FP32 accumulators. "
            "The old trace's version1 reference is rebound to seed version2 "
            "with operand dtype scope.",
            "fragment_layout,exact_once",
        ),
        (
            "physical_storage@1",
            "T16,T17,T18",
            "flash_fwd_kernel.h",
            167,
            174,
            "Tensor sV =",
            "Q/K/V shared storage and register statistics persist across their consumers; "
            "epilogue O aliases dead Q storage at lines443-469. Q/K sharing is "
            "disabled in this branch.",
            "lifetime,capacity",
        ),
        (
            "async_transfer@1",
            "T19,T20",
            "flash_fwd_kernel.h",
            309,
            346,
            "cp_async_wait<0>",
            "cp.async completion plus CTA visibility precede fragment reads. "
            "kernel_traits.h:124-133 supplies the concrete copy atom. Generic "
            "alignment alone is insufficient.",
            "read_after_complete",
        ),
        (
            "multi_buffer_rotation@2",
            "T21,T25",
            "flash_fwd_kernel.h",
            339,
            346,
            "__syncthreads",
            "Current K refill follows its QK last read within the iteration. "
            "Version2 adds late/private choices absent from the historical partial match. "
            "utils.h:150-158,174-180 fragment prefetch remains instruction-"
            "order binding, not a proved ring.",
            "readiness,lifetime; event-model instantiation remains open",
        ),
        (
            "operator_fusion@4",
            "T05,T06,T11",
            "flash_fwd_kernel.h",
            326,
            374,
            "gemm_rs",
            "QK, streaming normalization and PV share on-chip intermediates. "
            "Bind consumer_kind=contract,fused_ops=softmax_then_matmul,checkpoint=none and "
            "establish the rescalable consumer and lifetime premises; no "
            "external P output is requested.",
            "value_equivalence,lifetime; consumer-specific obligations retained",
        ),
        (
            "online_rescaled_reduction@2",
            "T06,T12",
            "softmax.h",
            137,
            185,
            "normalize_softmax_lse",
            "Source first-valid update initializes the frame; later updates "
            "rescale old numerator/denominator and finalization produces natural-log LSE. "
            "Map raw-score reference to natural scaled-logit frame with unit "
            "gain; positive final mass "
            "holds for visible equal-length causal rows, not padded query rows.",
            "invariant,finalize_equivalence,masked_prefixes,implementation_order",
        ),
        (
            "masked_tile_pruning@2",
            "T14,T15",
            "flash_fwd_kernel.h",
            90,
            101,
            "n_block_max",
            "Current n_block bounds and masked peeling at305-309 retain only "
            "identity-omitted keys. Rebind old single-edge fields to normalized"
            " leading/core/trailing "
            "metadata and descending traversal; this is a new binding, not "
            "version1 proof promotion.",
            "segment_partition,skipped_fully_masked,unmasked_really_unmasked,mask_applied",
        ),
        (
            "shared_resource_binding@1",
            "T23",
            "kernel_traits.h",
            70,
            100,
            "Swizzle<kSwizzle, 3, 3>",
            "Current Q/K/V and O maps compose a shared-memory swizzle. "
            "Bank mapping is an implementation binding; validate byte-"
            "coordinate preservation separately.",
            "value_preservation; native bank/layout binding open",
        ),
        (
            "operand_precision_conversion@1",
            "T09,T26",
            "flash_fwd_kernel.h",
            350,
            354,
            "convert_type<Element>(acc_s)",
            "FP32 score/probability fragment converts to FP16 for PV "
            "without converting FP32 normalization statistics. O converts at443. "
            "Current kernel's numerical acceptance depends on evaluator checks,"
            " not this source proposal.",
            "error_bound,convert_once",
        ),
        (
            "approximate_math_rewrite@1",
            "T08",
            "softmax.h",
            67,
            90,
            "exp2f",
            "Current exp2 scales raw scores and reference consistently with scale_log2; "
            "natural LSE uses scale_softmax and __logf at180. The base/unit "
            "mapping must be preserved.",
            "error_bound",
        ),
        (
            "cross_lane_reduction@1",
            "T07",
            "softmax.h",
            38,
            52,
            "Allreduce<4>",
            "Current row partials use a quad allreduce; utils.h:112-130 implements "
            "the shuffle tree. Full group participation and actual fragment row"
            " membership remain native obligations.",
            "group_complete,convergence",
        ),
        (
            "accumulator_operand_reuse@1",
            "T10",
            "flash_fwd_kernel.h",
            372,
            374,
            "convert_layout_acc_Aregs",
            "Current P fragment reinterprets the converted accumulator "
            "for PV A registers. utils.h:200-226 supplies the view; this witnesses the mechanism, "
            "not an automatic mapping for another compiler or MMA family.",
            "lane_layout_match",
        ),
        (
            "collective_fragment_load@1",
            "T24",
            "kernel_traits.h",
            40,
            44,
            "SmemCopyAtomTransposed",
            "Current source selects normal/transposed shared-memory "
            "collective loads for FP16 fragments. Reconcile actual CuTe atoms "
            "and lane/address maps "
            "against the chosen MMA; proposal domain guards alone do not "
            "establish native legality.",
            "lane_mapping,source_ready",
        ),
        (
            "staged_relayout@1",
            "T26",
            "flash_fwd_kernel.h",
            443,
            475,
            "Tensor sO",
            "Converted O is staged through shared storage and published before "
            "coalesced output stores. Shared aliasing follows Q last read; LSE "
            "is a separate direct output.",
            "write_before_read,no_overwrite_before_read,permutation",
        ),
        (
            "boundary_predication@1",
            "T22,T26",
            "flash_fwd_kernel.h",
            275,
            277,
            "copy",
            "Current K residue and V zero-fill have different policies at316-323. "
            "Output and LSE row guards at475-500 reject padded query rows. D128"
            " has no head-dimension "
            "tail; S513 still takes sequence guards. Untouched K needs exact later exclusion.",
            "no_oob_access,neutral_fill",
        ),
        (
            "vectorized_access@1",
            "T19,T26",
            "kernel_traits.h",
            124,
            146,
            "uint128_t",
            "Current copy maps provide wide aligned Q/K/V accesses and output stores. "
            "Vector and row guards must jointly cover actual BSHD strides; "
            "source width is not a fixed answer.",
            "coverage_exact_once,alignment",
        ),
    ]
    exposure = json.loads(
        (repo / "docs/runbooks/learndemo-1/controlled-replay/template-inputs.json").read_text()
    )
    inputs = {item["identity"]: item for item in exposure["entries"]}
    rows = []
    for identity, old, filename, first, last, witness, mapping, obligations in definitions:
        path = tree / filename
        lines = path.read_text().splitlines()
        if not any(witness in line for line in lines[first - 1 : last]):
            raise ValueError(
                f"Current source witness not found: {filename}:{first}-{last} {witness}"
            )
        rows.append(
            {
                "identity": identity,
                "historical_transactions": old.split(","),
                "current_source": str(path),
                "line_start": first,
                "line_end": last,
                "witness": witness,
                "mapping": mapping,
                "binding_checks": obligations,
                "obligations": [
                    item["name"]
                    for item in yaml.safe_load((repo / inputs[identity]["source"]).read_text())[
                        "obligations"
                    ]
                ],
                "status": "source_mechanism_witnessed_binding_obligations_retained",
                "outline": "absent"
                if identity
                in {
                    "operand_precision_conversion@1",
                    "approximate_math_rewrite@1",
                    "cross_lane_reduction@1",
                    "accumulator_operand_reuse@1",
                    "collective_fragment_load@1",
                    "staged_relayout@1",
                    "boundary_predication@1",
                    "vectorized_access@1",
                }
                else "retained_existing_canonical_outline",
            }
        )
    gaps = [
        {
            "id": "architecture_dispatch",
            "historical_transactions": ["T03"],
            "source": str(tree / "flash_fwd_launch_template.h"),
            "lines": [245, 274],
            "gap": "SM12.1 takes the non-sm8x non-dropout branch selecting128x64,four "
            "warps,Q-not-registers, "
            "no Q/K shared alias. The prescribed18 do not model dispatch "
            "profitability or compiler/ISA eligibility.",
        },
        {
            "id": "compiler_allocation_unrolling",
            "historical_transactions": ["T27"],
            "source": str(tree / "utils.h"),
            "lines": [139, 182],
            "gap": "Actual allocation, unrolling, occupancy and scheduled native "
            "instructions remain compiler outputs.",
        },
        {
            "id": "native_fragment_mapping",
            "historical_transactions": ["T10", "T23", "T24"],
            "source": str(tree / "kernel_traits.h"),
            "lines": [29, 146],
            "gap": "CuTe declarations witness a mapping but this reconciliation does "
            "not prove every native "
            "lane/register coordinate or another compiler's mapping. No proof "
            "outline is fabricated.",
        },
        {
            "id": "native_schedule_refinement",
            "historical_transactions": ["T20", "T21", "T25"],
            "source": str(tree / "flash_fwd_kernel.h"),
            "lines": [309, 435],
            "gap": "late/private variables explain why version1 partial status can be "
            "revisited; an actual "
            "event-model embedding and backend readiness/lifetime replay are still required.",
        },
    ]
    files = {
        name: hashlib.sha256((tree / name).read_bytes()).hexdigest()
        for name in sorted({row["current_source"].split("/")[-1] for row in rows})
    }
    covered = {t for row in rows for t in row["historical_transactions"]}
    explicit = {t for row in gaps for t in row["historical_transactions"]}
    assert covered | explicit == {f"T{i:02d}" for i in range(1, 28)}
    return {
        "scope": "evaluator-only; never included in implementation packages",
        "current_revision": source["revision"],
        "historical_revision": "a3a9dfa9bf10b52e55e8507b16ab0fe0dc1f38a0",
        "historical_decomposition": str(
            repo / "knowledge/seed/references/flash_attention_2_fwd/decomposition.md"
        ),
        "current_study": str(repo / "knowledge/studies/flash-attention/trace-cuda-fa2.md"),
        "revision_reconciliation": "Current retained source was inspected at each pinned witness. "
        "Historical transaction identities are a comparison index; byte "
        "equality across revisions is not assumed. "
        "Excluded variants remain excluded: "
        "dropout,GQA,paging,splitKV,softcap,ALiBi,window,backward.",
        "source_files": files,
        "selected_rule_count": 18,
        "historical_transactions_accounted": 27,
        "rule_rows": rows,
        "explicit_gaps": gaps,
    }


def render(mapping: dict) -> str:
    text = "# Evaluator-only source-to-rule reconciliation\n\n"
    text += mapping["revision_reconciliation"] + "\n\n"
    text += "The eighteen entries witness mechanisms, not a complete source explanation. "
    text += "Ten retain existing outlines; eight remain unproved source proposals. "
    text += "This map and original bindings are excluded from every public package.\n\n"
    for row in mapping["rule_rows"]:
        text += f"## {row['identity']} ({', '.join(row['historical_transactions'])})\n\n"
        text += f"Source: `{row['current_source']}:{row['line_start']}-{row['line_end']}`.\n\n"
        text += (
            row["mapping"]
            + f"\n\nObligations: {', '.join(row['obligations'])}. Outline: {row['outline']}.\n\n"
        )
    text += "## Explicit gaps\n\n"
    for gap in mapping["explicit_gaps"]:
        text += f"- {gap['id']}: {gap['gap']}\n"
    return text

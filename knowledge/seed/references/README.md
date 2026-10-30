# Reference decompositions (WP10a)

Hand decompositions of the first-batch reference kernels (docs/design/24 §4), produced with the
method of `src/kta/agents/skills/decompose-kernel/SKILL.md`. Each subdirectory is one Reference.

## Decomposition order (pool order)

The pool a reference is matched against is: the assumed seed (below) plus every Template and
Pattern proposed by the references decomposed before it. The order is fixed and is also the
order of the Template discovery curve:

1. `cutlass_ampere_tf32_gemm`
2. `flash_attention_2_fwd`
3. `flash_attention_3_fwd`
4. `marlin_int4_gemm`
5. `flashinfer_batch_decode`
6. `vllm_fused_moe`
7. `vllm_rms_norm`

## Files in each reference directory

Every YAML file holds the `body` of one record kind (spec/schemas.md), the same convention as a
family directory's `family.yaml` / `chain.yaml`. Refs are implied by the directory layout
(decision D100). The directory names `gaps/`, `templates/` and `patterns/` are the ones `scripts/gate`
uses to classify documents (D87); the decompose-kernel skill's `new_templates/` / `new_patterns/`
are renamed accordingly:

| File | Kind | Implied refs |
| --- | --- | --- |
| `source.yaml` | Source (fields not yet in spec/schemas.md, see D101) | — |
| `problem.yaml` | Problem (§4.8): the workload | — |
| `reference.yaml` | Reference (§4.5) | `source` = source.yaml, `problem` = problem.yaml |
| `chain.yaml` | Chain (§4.3) | `templates` = every `template` named in `items` |
| `reconstruction.yaml` | Reconstruction (§4.5) | `reference`, `chain`, `gaps` = `gaps/*.yaml` |
| `gaps/ux-NN-*.yaml` | Gap (§4.7) | — |
| `patterns/<name>.yaml` | Pattern (§4.1), proposed (D105) | — |
| `templates/<name>.yaml` | Template (§4.2), status proposed, assurance declared | `sources` = this directory's source.yaml |
| `decomposition.md` | not a record | human-readable workload, full transaction inventory with file:line, step-by-step matching argument |

Citations in `where`, `applies_to` and `decomposition.md` are `path:line` relative to the frozen
upstream commit in `source.yaml`.

## Counting rules (decision D102)

- A chain item is **matched** when its Template was in the pool before this reference and it
  expresses the step without a Gap. `Reconstruction.matched` counts those; `total` is the number
  of chain items.
- A chain item that uses a Template proposed by this reference, or that matches only partially,
  is not matched. Each such step, and each transaction that no chain item explains at all,
  is one entry in `unmatched_transactions`.
- `unmatched_transactions[i]` corresponds one-to-one to `gaps/ux-{i+1:02d}-*.yaml`; the Gap has
  the same `level` and `pattern_guess`, and the transaction description starts with `[ux-NN]`.
- A Gap whose step is resolved by a Template proposed in the same directory says so in its
  description. The Gap stays: it records that the pool could not express the step at the time.

## Assumed seed pool (decision D103)

WP4 owns `knowledge/seed/` patterns and the five paper GEMM Templates; they did not exist when
this WP was written. Chains bind against the following names, taken from paper.md §3.3 (table of
the five GEMM steps). If WP4 chooses different variable names, the bindings of these chain items
must be renamed (listed under "Needed changes" in deliveries/WP10a.md).

Patterns: `reduce`, `contract`, `map`, `store`, `transfer`, `sync`, `access`, `other`.

| Template | refines → produces | Variables |
| --- | --- | --- |
| `reduction_partition@1` | reduce → reduce | `num_groups`, `group_size`, `assignment` (which reduction indices go to which group) |
| `physical_storage@1` | store → store | `level` (reg, smem, tmem, gmem), `buffer`, `share_order` (order of values sharing one buffer) |
| `async_transfer@1` | transfer → transfer, sync | `mechanism` (software, instruction, controller), `issue_point`, `complete_point`, `granularity_bytes` |
| `multi_buffer_rotation@1` | store → store, sync | `S` (slots), `P` (prefetch distance), `beta` (trailing barrier, 0/1); obligation P ≤ S − 2 + β |
| `shared_resource_binding@1` | access → access | `binding` (request → bank/port map, e.g. an XOR swizzle), `queue_depth` (hardware; fixed on GB10) |

## Reproduction status

No reference was built or run on GB10 in WP10a (allowed by spec/workplan.md WP10a). `runnable_on`
records ISA eligibility only (decision D104): `["gb10"]` when every instruction the reference
needs exists on sm_121 per its documentation; `[]` (analysis-only) otherwise.

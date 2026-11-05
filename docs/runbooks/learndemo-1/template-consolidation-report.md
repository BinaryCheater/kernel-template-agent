# Template library consolidation

The first file-library revision is available at
[knowledge/library/README.md](../../../knowledge/library/README.md). It covers every
current seed, reference-package and study candidate without changing their
contracts. GEMM and causal-attention guides now give local binding examples that
can be used without opening the original study traces.

## Inventory and dispositions

| Item | Count |
|---|---:|
| Candidate source files before consolidation | 344 |
| Distinct source name/version identities | 343 |
| Canonical paired entries after consolidation | 312 |
| Source files withheld for missing outlines | 30 |
| Source files withheld for one conflicting identity | 2 |
| New outlines for unchanged contracts | 1 |
| Semantic merges / contract splits | 0 / 0 |
| Guides with separately curated case-relevant explanations | 11 |

Every source file has one row in
[mapping.json](../../../knowledge/library/mapping.json), including its original
identity, retained path, original proof path when present, disposition and canonical
destination. Original Template bytes and existing outline bytes are copied
unchanged. No old file is deleted or rewritten. Versions remain separately
addressable; no largest-version precedence is introduced. Discovery excludes
inputs, pool-records and snapshot directories, plus the generated library itself.

The 30 incomplete proposals remain outside the default catalog. They include
legacy reference proposals for fusion, fragment loads, boundary predicates,
asynchronous role schedules and online normalization. Later paired versions can
be selected explicitly where their own premises hold. Withheld proposals and
reasons are listed in [unresolved.md](../../../knowledge/library/unresolved.md).

The only same-identity collision is
`chunked_decayed_outer_product_state@2`: Mamba permits a broad chunk-width range
and partial tails; FlashInfer declares a narrower physical-block menu and adds a
clipped-subrange domain with sequence-boundary obligations. These are different
contracts, not formatting duplicates. Both source proposals are retained, neither
is published under that identity. Reconciliation must map every old variable and
obligation and assign a fresh version where needed; no precedence or union scope
has been inferred.

## Semantic overlap across different names

The following are explicit **retained independent/variant** dispositions. Shared
words or source architectures were not used as evidence of equivalence.

| Potential overlap | Retained distinction and reason against merging |
|---|---|
| `shared_memory_tiling@1` versus `output_tiling@2`, `reduction_partition@1`, `physical_storage@1`, `multi_buffer_rotation@2` | The first is a coupled symbolic GEMM model with fixed four-byte elements, fixed chip budgets and a specific barrier schedule. The separate rules expose independently selectable output ownership, reduction partition, storage and generation reuse. Erasing those differences would broaden the teaching model and lose its proof premises. Keep it as an alternative, with a guide that explains the boundary. |
| `async_transfer@1`, `multi_buffer_rotation@2` and `load_compute_store_role_pipeline@1` | Transfer readiness, destructive slot reuse and separate load/compute/store roles are different choices. An input ring and output ring can have different depths and completion events. A legal copy does not prove safe reuse, and a CTA-barrier lifetime inequality cannot certify a TMA role protocol. Retain each local contract. |
| `masked_tile_pruning@2`, `output_prefix_work_filter@1` and `empty_work_pipeline_projection@1` | Identity-only key-tile pruning, filtering invisible output work units and projecting no-work communication epochs act on different domains. Removing arithmetic does not remove a consumer's expected handshake. Keep the geometry and protocol rules separate. |
| Humming/QuACK/SGLang `bitstream_packing@1/@2/@3` versus FlashInfer `segmented_bit_packing@1` | The versions describe contiguous logical bit intervals with different writer/container support. Segmented packing rounds every independent segment to whole bytes and resets the bit position, so adjacent short segments must not share a byte. That changes offsets and ownership, not merely the implementation instruction. Retain versions and the segmented variant. |
| Fuser `persistent_value_rematerialization@1`, SGLang `statistic_dependent_rematerialization@1` and FlashAttention `softmax_probability_recomputation@1` | All exchange saved values for recomputation. Their retained frontiers differ: immutable producer inputs, a joined full-row statistic, or a row log-normalizer plus unchanged Q/K and mask/scale. The probability rule additionally carries log-base and checkpoint error obligations. A common umbrella would conceal the distinct consumer interfaces; retain the independently actionable contracts. |
| `tensor_axis_permutation@1`, `blocked_thread_register_layout@1` and `thread_ownership_exchange@1` | Descriptor stride permutation, lane/register ownership and an explicit shared-memory ownership exchange have different physical effects. A view can avoid a copy only when the consumer accepts the new strides; a lane exchange requires synchronization. Retain all three choices. |

This is a targeted semantic reconciliation pass over the current proposals, not a
claim that the library has no other possible equivalences. No semantic aliases
were asserted without a bidirectional binding/obligation mapping. New source
studies are outside this consolidation assignment.

## Concrete guide improvements

- `output_tiling@2` derives a 4-by-16 grid for a 127-by-1000 output and identifies
  the exact final CTA row/column ranges and suppressed stores.
- `reduction_partition@1` handles K=769 with 25 groups of 32: only one element of
  the final group is real. Sequential accumulation is distinguished from split-K.
- `compute_instruction_selection@2` binds an FP16/FP32 m16n8k16 instruction and
  derives 32 instructions for a 32-by-64-by-32 local contraction. It explains that
  row-major B still needs the correct fragment mapping and FP16 accumulation is
  outside the replay task. Its previously missing outline is explicitly authored
  during consolidation, with fragment ABI, target, padded partition and numerical
  premises; no proof result is claimed.
- `multi_buffer_rotation@2` shows S=2, P=1, beta=1 and the accepted lifetime
  inequality, then shows why dropping the barrier rejects that binding. It keeps
  slot generation, valid tail length and native asynchronous protocols separate.
- `online_rescaled_reduction@2` binds the ordinary running-max, unit-gain case and
  gives an independent scalar witness: scores 0 and ln(2), values 2 and 4 yield
  numerator 5, denominator 1.5 and output 10/3. It explains first-valid-tile initialization, masked prefixes, old-state correction and base2
  conversion. A second witness shifts both scores by -1000 and rejects the
  arbitrary-zero reference that would underflow all weights.
- `masked_tile_pruning@2` derives the kept interval and masked/core classification
  for equal-length causal attention. With S=513 it handles both an interior query
  tile and the single final visible row, including key and query padding.

All 312 guides retain the full variable domains, scope, preconditions, constraints,
transformation, backward mapping, effects, obligations and paired outline link.
Eleven receive hand-curated explanations and concrete examples; the remaining 301
are generated local binding recipes from the original metadata and retain the
original proposal's technical specificity. No whole-kernel implementation or
production answer chain is embedded in the guides. Observed source choices,
abstract domains and unimplemented extensions are distinguished explicitly.

## Reproducibility and checks

Generator: `tools/template_library.py`; curated input:
`knowledge/library/curation.yaml`; authored outline:
`knowledge/library/curated-proofs/compute_instruction_selection@2.yaml`.

Commands run:

```text
.venv/bin/python tools/template_library.py --check
candidate_files: 344
source_identities: 343
canonical_entries: 312
withheld_incomplete: 30
canonical: 312
unresolved_conflict: 2
Checked 940 files; 971 links

.venv/bin/ruff check tools/template_library.py tools/check_template_library_examples.py
All checks passed!

.venv/bin/python tools/check_template_library_examples.py
PASS 127x1000 output ownership
PASS K=769 tail partition
PASS instruction product count
PASS buffer lifetime accepted/rejected bindings
PASS fixed-model shared storage bytes
PASS all causal tile boundaries for S=513
PASS online common-frame scalar recurrence and negative-offset initialization
```

The regeneration check reloads every candidate through the current Template schema,
validates every published ProofOutline and its versioned target, verifies that
outline models name declared obligations, compares all generated files and checks
relative menu/guide links. Unique canonical identities follow from the grouped
identity map; mismatched duplicates are withheld. An additional check of all
current library Markdown found 975 resolvable relative links including README.
An independent retention pass checked all 344 source paths and hashes, all 312
unique canonical identities and byte-identical Template/outline copies, and 978
relative Markdown links across the library and report.
The local example checker enumerates output ownership and every causal boundary
for the odd-length example and verifies the documented arithmetic witnesses.
These are file integrity and local mathematical checks. No GPU, model run, Store
admission, ProofSummary or WP gate was performed for this publication.

## Independent guide correction before attention replay

The acceptance review identified a finite-arithmetic gap in the initial guide:
an arbitrary finite initial reference (for example zero) is mathematically valid
but underflows all weights when every FP32 score is near -1000. The guide now
leaves the reference inactive across masked prefixes and initializes it from the
actual maximum of the first valid tile, with no old-state correction. A translated
negative-score witness verifies the same 10/3 output and demonstrates the failed
zero-reference alternative. This clarifies the existing initial-state and positive
finite denominator premises; template.yaml and proof.yaml are unchanged.

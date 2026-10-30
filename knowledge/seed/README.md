# Seed knowledge

Imported by `kta init` (`kta.knowledge.import_seed`, decision D204). Every YAML file holds the
`body` of one record; refs are implied by the layout. Records are status `proposed`, assurance
`declared`, origin `real`.

| Directory | Kind | Content | Implied refs |
| --- | --- | --- | --- |
| `sources/` | Source | the "Learning to Formulate" paper at the upstream commit | — |
| `patterns/` | Pattern | reduce, contract, map, store, transfer, sync, access, other (docs/design/22 §2.2) | — |
| `structures/` | Structure | the nine exploitable structures of the paper's structure catalog | — |
| `templates/` | Template | the paper's five GEMM steps (below) | `sources` = every seed Source |
| `hardware/` | HwArch, HwModel | `arch_sm12x.yaml` (the GB10 architecture class) and `model_gb10.yaml` (its chip instance; documented and default values with per-value provenance, no KTA measurement yet; D463, D467) | HwModel `arch` = the class record |
| `references/` | several | hand decompositions of reference kernels (WP10a; see its README) | by layout (D100) |
| `families/` | Family packages | executable families (WP7) | — |

## The five GEMM-step Templates

| Template | refines → produces | Variables (X) | Key constraint / precondition |
| --- | --- | --- | --- |
| `reduction_partition@1` | reduce → reduce | `num_groups`, `group_size`, `assignment` | groups cover the reduction: `num_groups * group_size >= p.length` |
| `physical_storage@1` | store → store, transfer | `level`, `buffer`, `share_order` | capacity; lifetime `p.last_read + p.delta <= p.next_write` |
| `async_transfer@1` | transfer → transfer, sync | `mechanism`, `issue_point`, `complete_point`, `granularity_bytes` | aligned, whole copy operations |
| `multi_buffer_rotation@1` | store → store, sync | `S`, `P`, `beta` | `P <= S - 2 + beta` (C and Pre, K1) and `S * p.size_bytes <= p.capacity_bytes` |
| `shared_resource_binding@1` | access → access | `binding`, `queue_depth` (hardware) | on GB10 `queue_depth == hw.smem_queue_depth` (fixed chip variable) |

Variable names follow the assumed seed of WP10a (D103), so its reconstruction chains bind
unchanged. `p.<name>` are parameters of the refined transaction (D205); a chain item may bind
them. Obligations are listed with their planned check method; none is checked yet.

Limitations (decisions D207, D216): expressions cannot compare strings, so conditions on
string-valued variables (for example "mechanism is instruction needs cp.async") are stated in
obligations and Γ text, and `requires_hw` is empty for all five Templates.

Effects (WP15, D460, D468): every seed Template declares `effects` (transactions, storage,
overlap) instead of `cost`; `scripts/migrations/WP15_effects.py` performed the migration in
place (versions kept) and can be re-run (it is idempotent). Each physical transaction is
declared once, by the step that performs it: operand traffic by `async_transfer@1`, the
contraction by `compute_instruction_selection@2`, compulsory DRAM traffic and launches by
`output_tiling@2`, shared-memory accesses by `shared_resource_binding@1`, barriers and the
load/compute overlap by `multi_buffer_rotation@1`, storage by `physical_storage@1`.


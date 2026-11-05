# two_family_cta_pool@1: two family one shot cta claim pool

## When to consider

When independent prefill and decode grids should share one launch, give each CTA a domain-local family preference and claim one unique global job ticket. If that family is exhausted, try the other once. At least N0+N1 CTAs cover all jobs regardless of the preference sequence. Reuse counters only after grid completion and verify the unified CTA's barrier/thread contract. This controls preferred placement, not a guaranteed runtime share or speedup.

## Computation before and after

Assign one of two heterogeneous CTA bodies using per-domain weighted preferences and unique global tickets, with one overflow fallback.

Allocate one arrival counter for every allowed local-domain ID and two global family ticket counters. Zero them once for the invocation. Launch at least N0+N1 CTAs with threads/shared memory large enough for either body. Each CTA leader reads its domain ID and increments that domain's arrival counter. If N0<=N1, cycle through floor(N1/N0)+1 tags, preferring family0 on tag0 and family1 otherwise. If N0>N1, cycle through floor(N0/N1)+1 tags, preferring family0 on all but the last tag. Atomically claim a ticket from the preferred family. If it is out of range, claim exactly once from the other family. Broadcast the final family/ticket, join every reader, and only then reuse control scratch for the selected body. Execute that valid job once or exit uniformly if the final ticket is out of range. A CTA does not loop to another job. Preference is a placement heuristic; overflow counters count failed attempts and are not counts of completed work. Wait for whole- grid completion before consuming outputs or resetting shared counters.

Recover the original computation: Discard preference and failed ticket attempts; each successful family/ticket pair unrolls to exactly the corresponding CTA of the original two separate work grids.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|

## Coupled constraints and scope

Two nonempty independent CTA job families with one selected job per CTA, local-domain ratio preference and one fallback. Source POD uses physical SM IDs, separate prefill/decode counters and a combined launch. A sparse SM-ID namespace must fit the allocated span; number of active SMs is not a proof of that bound. Unequal thread-count bodies require a verified participation binding, not early thread exits before full-CTA barriers. Zero-family, concurrent static-counter reuse and persistent worker loops are excluded.

Scope predicate: `p.each_family_job_map_exact_and_immutable and p.families_independent_outputs_or_explicit_downstream_combine and p.domain_and_family_counters_zeroed_once_per_invocation and p.counter_storage_exclusive_until_grid_completion and p.observed_domain_ids_within_allocated_span and p.ticket_and_domain_increments_atomic and p.uniform_control_broadcast_before_scratch_reuse and p.invalid_tickets_exit_before_job_access and p.selected_family_barrier_participation_matches_unified_cta and p.shared_resource_lifetimes_valid and p.final_consumers_wait_for_grid_completion`.

- Precondition `pre_jobs` (semantic): `p.each_family_job_map_exact_and_immutable and p.families_independent_outputs_or_explicit_downstream_combine` — Each valid family ticket maps to exactly one original job; outputs are independent or combined afterward under an explicit contract.
- Precondition `pre_state` (implementation): `p.domain_and_family_counters_zeroed_once_per_invocation and p.counter_storage_exclusive_until_grid_completion and p.observed_domain_ids_within_allocated_span` — Counter storage belongs to this invocation, all observed domain IDs are in range, and reset/reuse cannot overlap a live grid.
- Precondition `pre_atomic` (implementation): `p.ticket_and_domain_increments_atomic and p.uniform_control_broadcast_before_scratch_reuse and p.invalid_tickets_exit_before_job_access` — A leader claims tickets atomically, broadcasts one decision to all CTA roles, and invalid final tickets cannot access a job.
- Precondition `pre_barriers` (implementation): `p.selected_family_barrier_participation_matches_unified_cta and p.shared_resource_lifetimes_valid and p.final_consumers_wait_for_grid_completion` — Selected device bodies have legal barrier participation for the unified launch, use shared storage safely and complete before downstream consumers.
- Constraint `c_domains` (implementation): `p.domain_counter_span >= 1` — Allocate a nonempty counter span covering every observed local-domain ID.
- Constraint `c_nonempty` (semantic): `p.family0_jobs >= 1 and p.family1_jobs >= 1` — Both families are nonempty, so the ratio selector never divides by zero.
- Constraint `c_grid` (implementation): `p.launch_ctas >= p.family0_jobs + p.family1_jobs and p.launch_ctas < 2147483647` — At least one CTA per valid job is available and signed source counters do not overflow.
- Constraint `c_resources` (resource): `p.launch_threads >= max(p.family0_threads, p.family1_threads) and p.launch_smem_bytes >= max(p.family0_smem_bytes, p.family1_smem_bytes) and p.launch_smem_bytes >= 8` — One selected family executes per CTA, so resource envelopes use the maximum plus space for the initial two-word control broadcast.

## Local example and misuse

For a local instance of `work_assign`, identify the object `two_family_one_shot_cta_claim_pool` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/flashinfer/delta/two_family_cta_pool@1/template.yaml`. Original outline: `knowledge/studies/flashinfer/delta/two_family_cta_pool@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: A single launch colocates independent work families while reusing a maximum-sized CTA resource envelope.. Storage: Per-domain arrival counters and two family ticket counters; body scratch is time-shared after control broadcast.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage:
- id: claim_counters
  level: dram
  scope: kernel
  bytes: 4 * (p.domain_counter_span + 2)
overlap: []
```

## Obligations and numerical checks

- `exactly_once_and_coverage` (proof): Atomic successful tickets are unique, every valid job is claimed with the grid bound, and failed claims never execute a job.
- `counter_lifetime` (test): Every invocation has zeroed exclusive counters with valid domain indexing until all claims and computations finish.
- `uniform_dispatch_and_barriers` (test): All CTA roles receive one family/ticket and execute only a device body whose barrier participation and resource envelope match the launch.
- `completion` (test): Counter exhaustion means claimed work rather than completed work; downstream consumers wait for actual grid completion.

Independent output ownership preserves each body's arithmetic; any subsequent split-state merge retains its own numerical policy and completion contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).

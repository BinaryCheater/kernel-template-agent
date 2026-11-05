# output_prefix_work_filter@1: prefix valid output work unit

## When to consider

When grouped output buffers have allocated row capacity but only a prefix is visible, consider filtering whole cooperative work units before their pipelines. Preserve every visible output once and let all peers and roles follow the same retained sequence; bind unit_rows to the cluster footprint, not one CTA. With128-row units and valid_rows1, retain the first unit and skip starts128 onward; partial units still write padding, and the pinned source needs a validity guard before terminal metadata lookup.

## Computation before and after

Remove cooperative output work wholly outside immutable valid row prefixes.

Enumerate each worker's base sequence of cooperative output units. Check validity before reading a unit's group prefix. Except for an optionally bypassed initial unit, compute the first row of the whole cooperating unit and keep it iff that row is less than the group's valid row count. Skip empty units without advancing their compute/input/output pipelines; publish the same next retained unit or terminal marker to every role. Retained units preserve the original coordinate mapping and output arithmetic, including any allocated unobservable padding.

Recover the original computation: Reinsert the omitted units whose row intervals are wholly outside the visible output prefix; these units have no visible outputs.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| unit_rows | compile / boundary | {"lo": 1, "hi": 65536} | Rows covered by the entire cooperating work unit rather than one member CTA. |
| bypass_initial | compile / software | [false, true] | Allow each worker's first valid unit to execute without filtering; padding remains unobservable and allocated. |

## Coupled constraints and scope

Output rows valid on [0,valid_rows) separately for each group, with disjoint cooperative row units. This is not element masking: a unit crossing the boundary still executes padded rows. The pinned masked grouped example filters subsequent units using the cluster's first row and bypasses initial units. Its scheduler reads group metadata after advancing to possibly invalid work, so validity-first guarding is required before a backend binding satisfies this rule. Non-prefix masks or observable padding writes need a different implementation or variant.

Scope predicate: `p.prefix_bounds_legal and p.base_work_partition_exact and p.prefix_metadata_immutable and p.padding_accesses_safe and p.padding_writes_unobserved and p.invalid_work_checked_before_metadata and p.peers_share_filter_result and p.roles_share_kept_sequence`.

- Precondition `pre_domain` (semantic): `p.base_work_partition_exact and p.prefix_metadata_immutable` — Base work visits each valid output exactly once and prefix metadata stays fixed.
- Precondition `pre_padding` (semantic): `p.padding_accesses_safe and p.padding_writes_unobserved` — Retained partial or initial units may execute and store padding without changing a visible result.
- Precondition `pre_control` (implementation): `p.invalid_work_checked_before_metadata and p.peers_share_filter_result and p.roles_share_kept_sequence` — Only valid work reads metadata; all cooperating peers and pipeline roles agree on the retained units.
- Constraint `c_unit` (semantic): `unit_rows == p.cooperative_unit_rows and p.prefix_bounds_legal` — Prefix lengths are within allocated row capacities and the unit covers all cooperating peers.

## Local example and misuse

For a local instance of `work_assign`, identify the object `prefix_valid_output_work_unit` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cutlass/delta/output_prefix_work_filter@1/template.yaml`. Original outline: `knowledge/studies/cutlass/delta/output_prefix_work_filter@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Prefix metadata and a shared retained-work sequence.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `visible_coverage` (exhaustive): Every visible output coordinate is retained exactly once and no skipped unit intersects the valid row prefix.
- `cooperative_agreement` (test): Peers classify their whole cooperative unit identically and every pipeline role consumes the same retained sequence.
- `bounded_metadata` (test): Terminal or invalid work does not read nonexistent prefix metadata.
- `padding_contract` (test): Extra rows from a partial or bypassed initial unit are allocated and have no observable output effect.

Retained output arithmetic is unchanged; padding and output observability are part of the contract.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).

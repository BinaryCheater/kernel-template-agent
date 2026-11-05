# adaptive_key_partition_membership@1: adaptive key partition membership

## When to consider

When an exact integer set varies in local density, partition keys by a high-bit prefix and choose sorted low-key arrays or fixed bitsets per partition, preserving the serialized container contract. Key0x12340007 selects high0x1234,low7. A run(start5,length2) includes5,6,7; using length as an exclusive end wrongly omits7. Cardinality4096 array and4097 bitset must return the same exact set semantics.

## Computation before and after

When an exact integer set varies in local density, partition keys by a high-bit prefix and choose sorted low-key arrays or fixed bitsets per partition, preserving the serialized container contract.

Split a32-bit key into high16 container key and low16 member. Locate high key by linear search for fewer than8 containers, otherwise binary search. Without a run tag, decode cardinality as stored_card+1; cardinality<=4096 uses sorted uint16 lows, otherwise an8192-byte bitset. Array membership uses linear or binary search, bitset uses word/bit test. Imported run containers test inclusive interval start<=low<=start+stored_length. Builder sorts/deduplicates as required, identifies container starts, scans payload sizes, schedules array/bitset construction and serializes offsets.64-bit membership first selects the high32 bucket then applies32-bit logic.

Recover the original computation: Erase auxiliary organization and recover the visible results under the stated contract.

## Bind the rule

1. Bind computation, extents, dtypes and layout to `p.*`; bind target facts to `hw.*` and workload facts to `case.*`. These are fixed.
2. Select values only from the declared variable domains below. A compile binding is a specialization point; a runtime binding is supplied when executing. Boundary roles must agree with the adjacent rule or caller.
3. Evaluate the scope, every precondition and every constraint. Supply evidence for semantic booleans; setting an unproved premise to True is insufficient.
4. Implement the declared mapping and discharge the obligations for this binding. Keep numerical acceptance separate from the exact-arithmetic argument.

| Variable | Binding / role | Declared domain | Meaning |
|---|---|---|---|
| low_bits | compile / software | [16] | Low-key domain size65536 in each32-bit Roaring container. |
| array_cutoff | compile / software | [4096] | Source format cardinality cutoff; changing it requires matching format metadata/consumer. |

## Coupled constraints and scope

NVIDIA Roaring bitmap32/64 membership. Native builder emits arrays/bitsets; run membership consumes imported serialized format rather than assuming this builder emits runs. Container search threshold8 is policy, array cutoff4096 is format-sensitive. Resource effects: Sparse containers cost2*card bytes, dense bitsets8192 bytes per16-bit domain; high-prefix directory and optional offsets add overhead. Different density choices change storage/read work without approximating membership.

Scope predicate: `p.sorted_unique_container_members and p.metadata_offsets_valid and p.container_tags_match_payload and p.input_width_contract and p.immutable_completed_build`.

- Precondition `pre_0` (semantic): `p.sorted_unique_container_members and p.metadata_offsets_valid` — Each container low-key set is sorted unique and serialized counts/offsets identify valid payload ranges.
- Precondition `pre_1` (semantic): `p.container_tags_match_payload and p.input_width_contract` — Array/bitset/run tags agree with actual encoding;32/64-bit key partition rules match producer and consumer.
- Precondition `pre_2` (semantic): `p.immutable_completed_build` — Membership reads a complete immutable bitmap or correctly ordered imported serialized buffer.

## Local example and misuse

For a local instance of `map`, identify the object `adaptive_key_partition_membership` and replace only that operation by the program above. Write down one binding for every variable in the table and the caller facts used by each predicate. This recipe is not a validated backend example. Reject out-of-scope bindings even if the variable domains accept them.

## Implementations, evidence and extension boundary

The contract above is retained from the proposal. The original source instance and any implementation-specific choices remain provenance; a declared domain does not establish that every value has an implemented backend. Before selecting an instruction or mapping, establish target and compiler support. Any new tail, numerical policy or mapping outside this contract is an extension requiring a revised rule/outline and implementation evidence.

Declared capabilities: `[]`. Declared probed targets: `[]`.

Original proposal: `knowledge/studies/cucollections/delta/adaptive_key_partition_membership@1/template.yaml`. Original outline: `knowledge/studies/cucollections/delta/adaptive_key_partition_membership@1/proof.yaml`. Publication here preserves a proposed contract; it does not add proof assurance.

## Expected benefit and cost

Reuse: not specified. Storage: Sparse containers cost2*card bytes, dense bitsets8192 bytes per16-bit domain; high-prefix directory and optional offsets add overhead. Different density choices change storage/read work without approximating membership.. Structured effects describe work and resource changes, not timings.

```yaml
transactions: []
storage: []
overlap: []
```

## Obligations and numerical checks

- `partition_identity` (proof): Recombining high prefix and low member recovers exactly the original integer without signed/truncation changes.
- `representation` (proof): Every container type denotes the same exact local set its metadata describes.
- `search_order` (proof): High keys/arrays/runs are ordered as required, with inclusive run ends and card-minus-one decoded correctly.
- `serialization` (proof): Aligned/misaligned loads and computed/stored offsets remain within the immutable payload and agree with format.

Preserve the supplied arithmetic contract; reassociation, quantization and approximate search require separate permission.

Read [the paired outline](proof.yaml) for assumptions and open checks. No ProofResult or ProofSummary is created by consolidation.

## Related rules

No relation is declared; composition still requires matching interfaces.

Structured contract: [template.yaml](template.yaml).

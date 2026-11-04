# Accumulating Templates from kernel source

A Template records one parameterized refinement of a computation. A reference instance records where the source uses it and which values it chooses.

## What the next kernel author needs

Use the existing fields. Write one independently usable design choice.

| Reader question | What to record |
|---|---|
| When would I select it? | Start the summary with the structural signal in the workload and current design. State the semantic invariant separately from the supported width, shape, architecture and tail behavior of the pinned implementation. Use scope and constraints for actual applicability; a retrieval cue may suggest a variant outside those bounds. |
| What can I change? | In variables and constraints, distinguish a free choice (including selection of the rule itself), a value bound by the caller or another step, and a derived quantity. Give the units, legal domain and coupling. Label narrower source menus as such. |
| What do I write differently? | In program, state the before/after operation, inputs consumed, outputs or guarantees produced, and the binding or mapping that connects them. Link a Realisation when applying the rule needs a specific instruction sequence or layout. |
| What tradeoff am I buying? | In effects, describe how work, traffic, live storage or synchronization changes with the choice. Give formulas where known; keep measured speedups scoped to their evidence. |
| How do I use it correctly? | Include one concrete binding and one consequential misuse. In obligations and the paired outline, explain the invariant, assumptions and outstanding checks. Link the source instance. |

Retain a rule when it contributes a choice, a reusable mapping, or a non-obvious obligation that changes generated code. Start narrow when only one implementation is understood. Generalize on evidence. Record a routine coding step in the instance trace if it contributes no such knowledge.

## Writing rules

- **Name.** Use a stable lowercase snake_case name for the mechanism, such as `reduction_partition`. Keep repository names, kernel names, observed constants, and target names in the instance record. Reuse an existing name for the same rule. Version changes to its contract.
- **Granularity.** Describe one design object and the local correctness obligation it introduces. Split two choices when each has a distinct object and obligation. Keep choices together when they depend on the same invariant and have no clear intermediate contract.
- **Content.** State the input pattern, transformation, independent variables and domains, derived quantities, constraints, workload and hardware requirements, resource effects, and structural relations to other Templates. Give a reason for each relation. Record a suspected benefit on a workload or architecture as a scoped hypothesis.
- **Evidence.** Distinguish what the interface declares, what the code implies, what another variant demonstrates, and what the model infers. For a hard-coded value, explain what would have to change with it before proposing a variable domain.

## Short example

**Template — reduction_partition.** Replace one reduction with groups of partial sums followed by a combine. For contiguous groups of size four and ten inputs, the groups cover indices 0–3, 4–7, and 8–9. Four is the instance binding. Group size is the choice; group count in this example follows from the length and group size. Changing four to eight gives two groups under the same rule. [Complete Template YAML](../../../knowledge/seed/templates/reduction_partition.yaml)

**Proof outline — reduction_partition.** For positive group size S, input i belongs to group floor(i/S) at offset i mod S. This gives each valid index exactly one place. Guard the final group at the input length. Combining each completed partial once recovers the sum in exact arithmetic. The numerical contract must allow the rounding effects of reassociation. The index argument covers contiguous assignment; synchronization before the combine is a separate obligation. [Complete proof outline YAML](../../../knowledge/seed/proofs/reduction_partition.yaml)

## One pass, three outputs

1. **Instance decomposition.** Fix the source revision and starting pool. For a library, inventory its NVIDIA entry points and dispatch paths, then trace each mechanism group. Trace each design step to file and lines. Record branch conditions, actual bindings, and dependencies.
2. **Pool delta.** Match each step against existing Templates first. Different bindings add an instance; backend-specific code adds a Realisation. A new refinement adds a Template. Pair each new or revised rule with a ProofOutline for that version. For each obligation give the argument and its assumptions, or record the gap, next check, and when it must be resolved.
3. **Review.** Check references, schema, and constraints. Substitute source values and compare predicted branches, operations, and boundaries with the code. Correct once, then mark each step explained, partial, or open. Close the pass with the remaining questions. Zero new Templates is a valid result.

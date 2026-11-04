# Version1 to version2 — explicit projection and changed obligations

Version1 remains in the pinned seed pool. This proposal does not edit its body,
source instances or proofs. Version2 has the same mechanism name and state object,
with a selectable reference/gain frame; it is not an unrelated rule identity.

| Version1 field/obligation | Version2 owner | Projection or change |
|---|---|---|
| `state=max_sum/max_sum_acc` | Same field and domain | Carry reference/denominator and optional numerator. Legacy names remain; a lagged reference is not required to be the maximum. |
| `rescale_point` three values | Same field/domain plus `implementation_order` | Backend must correct only contributions from the old frame before combining; no new physical schedule is proved. |
| `finalize=deferred_divide/none` | Same field/domain | Normalize once, or expose reference/L/O and the additional gain together. |
| `traversal=ascending/descending` | Same field/domain, complete partition premise | Both orders have bounded recurrence witnesses. |
| `inf_guard=false/true` | Same field/domain, `masked_prefixes` | Final zero-mass rows are excluded in this version. Version1's broader masked-final-row obligation is not discharged or migrated automatically. |
| `pre_assoc` | `pre_partition` | Keep associativity/rescalability and add complete partition/no nonidentity pruning explicitly. |
| `invariant` exact prefix max plus sums | Framed `invariant` | With running_max/lag0/gain0 and first finite tile reference, recover version1's exact maximum and sums on the finite positive-mass domain. Lagged reference changes the invariant, not just its implementation. |
| `finalize_equivalence` | Same obligation plus gain/log compensation | Common gain cancels from O/L; LSE must subtract log(G). |
| `masked_rows` | `masked_prefixes` and positive finite final denominator guard | Deliberate narrower admitted domain; a final all-masked policy requires separate work. |
| Version1 transaction estimates | No universal estimate in version2 | Reference freezing changes actual correction incidence. Counts need realized reference updates/warp ballots and native lowering. No cost equivalence is claimed. |
| Source fast-math/rounding risk | `casts_and_math_admitted`, range guard | Real identities do not prove native FP8 probability casts or packed TMEM updates. |

A version1 algebraic projection sets `reference_policy=running_max`,
`threshold_log2=0`, `weight_gain_log2=0` and keeps every old variable unchanged.
The check validates all five old domains and both versions' premises on that
projection. This is a finite positive-mass semantic projection, not a claim that
all old capabilities or effects are equivalent in both directions.

The source binding uses positive softmax scale, lag8/gain1 or immediate max/gain256,
complete retained tiles, admitted range/rounding and matched state transitions.
Sink+gain256 currently fails the extra-term frame premise. Value-based pruning
also fails the no-dropped-terms premise. Both remain excluded until separately
corrected/bounded; their existing source instances are still addressable.

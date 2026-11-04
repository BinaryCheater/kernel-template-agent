# Draft-prefix acceptance and target residual correction

When a draft model proposes a token chain, the target can reuse a consecutive
prefix and correct the first rejection. This is a coupling between **two**
distributions, distinct from conditioning one distribution on a filtered support.
The new [candidate](delta/draft_target_residual_correction@1/template.yaml) and
[outline](delta/draft_target_residual_correction@1/proof.yaml) preserve that law,
its history requirements and its observable counters.

Route: `flashinfer/sampling.py:2326` -> sampling custom op ->
`include/flashinfer/sampling.cuh:1897-2114`. Paths refer to the retained
[tree](inputs/sources.json). The source names target probability `q` and draft
probability `p`; equations below use P=target and Q=draft.

| Decision / concrete source binding | Rule field or boundary | Evidence |
|---|---|---|
| One CTA/request, draft rows[L,V], target rows[L+1,V] | `pre_distributions`, `pre_history` | Target/draft rows must describe the same conditional histories and generation; draft IDs must actually come from Q. The extra target row supplies the bonus token |
| First loop accepts while u*Q[d]<P[d], otherwise sets first-rejected position r | `pre_acceptance`, `pre_prefix` | Ideal acceptance probability is min(1,P/Q); only the consecutive prefix is emitted. A later draft cannot bypass a rejected earlier history |
| Fresh acceptance rechecks for positions r..L-1, including r, add to a diagnostic counter | `diagnostic_mode=resample_unemitted` | Diagnostic count=r+passing rechecks; emitted-draft count=r. These may differ, and both source counters use+= rather than overwrite |
| Full vocabulary reduction of max(P_r-Q_r,0), then masked CDF using that same residual mass | `pre_residual` | Sample the positive **target-minus-draft** correction. A fresh draw from P or reversed subtraction is not the same coupling |
| After full acceptance, draft loads are omitted and p_vec stays zero at target row L | Bonus branch in `program` | Residual becomes P_L, so the one extra token is directly target-distributed |
| First rejected/bonus output is followed by-1 padding | `pre_prefix`, output capacityL+1 | The valid block is accepted drafts plus exactly one correction/bonus; diagnostic successes never populate the invalid tail |
| Sequential CURAND acceptance, diagnostic and final CDF draws | `pre_rng` | Any rejection consumes(r+1)+(L-r)+1=L+2 uniforms; all accepted consumesL+1. Removing diagnostic work shifts the correction RNG identity unless explicitly remapped |
| Strict comparison with CURAND u allowed to equal1; CDF fallback uses last eligible residual ID | Native endpoint/zero-residual boundary | At P=Q and u1, literal comparison rejects but every residual weight is zero. The candidate requires a matching endpoint/zero-mass policy; that premise is not established for this source path |

For one step, accepted token i has joint mass min(Q_i,P_i). Total rejection mass
is sum_i(P_i-Q_i)_+, and residual sampling contributes exactly(P_i-Q_i)_+ to i.
Their sum is P_i. Matching conditional histories extends the construction along
the accepted prefix. After a correction the next target evaluation must use the
corrected history, not rows conditioned on discarded draft suffix tokens.

The [exact-rational check](checks/check_draft_target_residual.json) covers387
small distribution pairs, including zero components/disjoint supports and25
equal-row cases. It also enumerates two conditional draft positions,10 positive-probability
variable output blocks, and a target continuation to a fixed three-token horizon:
all8 sequence probabilities equal the reference target chain. This avoids the
incorrect claim that conditioning on the emitted block length preserves the
unconditional target-prefix distribution.

At Q=[1/2,1/2], P=[1/4,3/4], replacing the residual correction with a target redraw
gives[5/16,11/16] instead of P. If the actual proposal generator is[3/4,1/4] while
the verifier still declares Q=[1/2,1/2], the output becomes[3/8,5/8]. Proposal
provenance is therefore a semantic requirement, not optional metadata.
A counter witness has zero emitted drafts but two successful independent
diagnostic rechecks. The source endpoint witness identifies an explicit native
boundary; no source repair or native probability guarantee is claimed.

`adaptive_support_rejection@1` supplies neither this target-minus-draft correction
nor history coupling. Reuse only the common masked-CDF implementation after
binding its residual weights, complete mass and finite endpoint policy. Native
FP32 residual cancellation, RNG/CTA convergence, counter exclusivity and output
publication remain realization obligations.

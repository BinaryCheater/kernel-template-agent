# Author correction pass

The source-study pass was reviewed against the pinned source, rule identities, group
inventory and source-to-rule interfaces after all thirteen family traces were written.

| Finding | Correction | Reverification |
|---|---|---|
| A matrix packer bijection alone did not establish its consumer convention | Compose legacy G2S, half-group S2R, nibble decode and the documented MMA B map | All 8192 B coordinates agree; wrong interleave rejected |
| The same logical coordinate had two apparent word addresses | Distinguish N64 stride (word 277) from N128 stride (word 533), retaining nibble 5 | Repack and composition witnesses carry their separate shapes |
| WGMMA logical operands were easy to confuse with PTX A/B | Record the source wrapper's A/B and M/N swap explicitly | Real source generates m64n64k16 for the logical64x64 example; full lane map stays open |
| Hadamard helper scope exceeded the public input context | Limit the proposal H domain to512; H remains a workload boundary parameter | Added all H 512 basis vectors, including cross-warp stages |
| Affine quantization grouping/type choices could silently change the workload | Require group size, bit width and zero-point kind to match the numerical contract | Matching witness accepted; changing the requested zero-point kind rejected |
| SM90 local branch could be preempted or fitted afterward | Execute actual upstream early exits and final tile fitting; retain seed candidate scores | M32 MMA to M33 WGMMA boundary recovered at fixed format/device |
| Atomic combine was not equivalent to ordered lock accumulation | Introduce a separate first-writer/countdown object |873 completion orders and two generations; premature completion mutation exposes reuse failure |
| Source identity and branch presence could be mistaken for full correctness | Give every decision a disposition, preserve branch anchors as partial and attach next evidence |53 rows reconciled,15 Gap bodies schema-valid; no branch proof percentage reported |

All study checks and Ruff passed after these corrections. The four proposals retain declared
assurance. None has a registered checker model or a ProofResult; each outline states which
premises its arithmetic witness covers and names the next evidence for unresolved obligations.

No existing rule was mutated, so no old bindings or proof certificates were silently migrated.
The starting repository snapshot and reviewed index were checked; no sibling proposal became
a dependency. The public source continues to contain all twelve global entries accounted for
by the thirteen mechanism groups; shared device bodies are not counted as separate kernels.

The repository static gate exposed a separate pre-existing acceptance-lock mismatch, retained
in `static-gate.txt` and `gate-baseline.json`. No acceptance lock, test, runtime implementation,
old delivery or unrelated performance-model work was changed by this study.

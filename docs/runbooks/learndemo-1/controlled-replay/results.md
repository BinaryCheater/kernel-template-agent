# Source attention control: initial confirmation

This page retains the initial confirmation and its raw evidence. See [final findings](/Users/dingyi/Projects/agentic-codesign/kta/docs/runbooks/learndemo-1/controlled-replay/conclusion.md) for the uniformly applied frozen-source measurement recovery, execution problems, and final interpretation.

All nine outcomes recorded; 2/9 confirmed target outcomes. Full matched outcomes are required before assessing supplied-knowledge benefit.

## Original comparator and computation

Unchanged CUDA FlashAttention-2 dense causal FP16 D128 forward, with a declared thin allocation/parameter/launch host binding. Both entries compute O and natural-log LSE. Q/K/V are BSHD [1,S,8,128], S=512,2048,513; O is FP16 BSHD and LSE is FP32 BHS. GB10 sm_121; primary timing includes the complete GPU entry under graph replay. The target permits about 11.1% more time on every shape.

[Pinned source/build inventory](</Users/dingyi/kta-data/dev/source-attention-control-v1/preparation/baseline/remote-retained/source-inventory.json>), [release build recipe](</Users/dingyi/kta-data/dev/source-attention-control-v1/preparation/baseline/remote-retained/build-recipe.json>), [original output checks and runtime identity](</Users/dingyi/kta-data/dev/source-attention-control-v1/preparation/baseline/remote-retained/p1-result.json>), [Source locations / rule map](</Users/dingyi/kta-data/dev/source-attention-control-v1/preparation/source-template-map.md>).

Observed source traits: D128, 128 query rows, 64 keys, 4 warps, Q-register flag=false, Q/K alias flag=false.

## Every scheduled session

| Slot | Arm | Final O/LSE | Target | Active | GPU wait | Session wall | Submissions | Outcome |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run-01 | self-selection | pass | not attained | 1,257.46 s | 1,865.85 s | 3,123.31 s | 15 | correct; target not attained |
| run-02 | prescribed | pass | pending/unresolved | 1,800.81 s | 1,545.54 s | 3,346.35 s | 16 | measurement invalid/unresolved |
| run-03 | no-template | pass | pending/unresolved | 750.81 s | 1,062.38 s | 1,813.19 s | 7 | measurement invalid/unresolved |
| run-04 | prescribed | pass | not attained | 1,337.33 s | 1,499.39 s | 2,836.72 s | 6 | correct; target not attained |
| run-05 | no-template | pass | pending/unresolved | 813.92 s | 844.76 s | 1,658.68 s | 10 | measurement invalid/unresolved |
| run-06 | self-selection | pass | attained | 1,273.86 s | 2,335.31 s | 3,609.17 s | 16 | target attained |
| run-07 | no-template | pass | attained | 1,128.55 s | 2,109.94 s | 3,238.49 s | 12 | target attained |
| run-08 | self-selection | pass | pending/unresolved | 1,557.65 s | 1,733.11 s | 3,290.76 s | 16 | measurement invalid/unresolved |
| run-09 | prescribed | pass | not attained | 1,035.99 s | 1,636.20 s | 2,672.19 s | 13 | correct; target not attained |

Session wall is accumulated active time plus verified GPU waiting; evaluator time is not added again. Confirmation worker time is separate.

## Confirmed complete-entry performance

| Slot / S | Original us | Generated us | Generated uses | Window median ranges us |
| --- | --- | --- | --- | --- |
| run-01 / 512 | 22.204 | 22.541 | 1.52% more time | original 21.888–22.336; generated 22.539–22.546 |
| run-01 / 2048 | 148.179 | 161.113 | 8.73% more time | original 148.075–150.770; generated 160.956–162.224 |
| run-01 / 513 | 23.620 | 26.639 | 12.78% more time | original 23.603–23.643; generated 26.633–26.665 |
| run-04 / 512 | 21.675 | 22.537 | 3.98% more time | original 21.609–22.212; generated 22.534–22.539 |
| run-04 / 2048 | 148.579 | 161.247 | 8.53% more time | original 148.088–148.642; generated 160.632–161.661 |
| run-04 / 513 | 23.687 | 26.456 | 11.69% more time | original 23.615–24.548; generated 26.324–26.547 |
| run-06 / 512 | 22.286 | 22.511 | 1.01% more time | original 22.276–22.288; generated 22.451–22.512 |
| run-06 / 2048 | 148.259 | 159.831 | 7.81% more time | original 148.177–148.335; generated 159.812–159.874 |
| run-06 / 513 | 23.636 | 24.842 | 5.10% more time | original 23.611–24.406; generated 24.620–25.617 |
| run-07 / 512 | 22.112 | 22.493 | 1.72% more time | original 21.758–22.286; generated 22.338–22.524 |
| run-07 / 2048 | 148.146 | 157.951 | 6.62% more time | original 148.069–152.852; generated 157.906–158.105 |
| run-07 / 513 | 23.772 | 23.651 | 0.51% less time | original 23.585–24.481; generated 23.548–24.019 |
| run-09 / 512 | 22.144 | 22.491 | 1.57% more time | original 22.074–22.190; generated 22.420–22.515 |
| run-09 / 2048 | 148.815 | 158.998 | 6.84% more time | original 148.178–149.212; generated 158.896–159.718 |
| run-09 / 513 | 23.585 | 26.568 | 12.65% more time | original 23.576–24.347; generated 26.550–26.640 |

Only correct candidates with valid three-window confirmation enter this table. All failures/invalid samples remain linked below; eager diagnostics are separate.

## Native usage and retained-price estimates

| Slot | Input incl. cache | Cached | Uncached | Cache write | Output | Reasoning subset | API-equivalent USD |
| --- | --- | --- | --- | --- | --- | --- | --- |
| run-01 | 10,522,982 | 10,368,384 | 154,598 | 0 | 15,978 | 3,293 | 1.51 |
| run-02 | 13,514,367 | 13,422,720 | 91,647 | 0 | 27,090 | 8,493 | 1.80 |
| run-03 | 4,009,396 | 3,904,000 | 105,396 | 0 | 10,284 | 2,519 | 0.70 |
| run-04 | 6,614,882 | 6,539,904 | 74,978 | 0 | 11,168 | 1,743 | 0.92 |
| run-05 | 2,964,337 | 2,876,032 | 88,305 | 0 | 10,282 | 1,944 | 0.57 |
| run-06 | 12,065,262 | 11,985,920 | 79,342 | 0 | 20,434 | 4,852 | 1.56 |
| run-07 | 8,087,334 | 7,981,312 | 106,022 | 0 | 17,091 | 4,691 | 1.18 |
| run-08 | 14,200,136 | 14,113,792 | 86,344 | 0 | 19,217 | 4,466 | 1.78 |
| run-09 | 8,999,006 | 8,842,752 | 156,254 | 0 | 16,829 | 3,844 | 1.37 |

Actual billing is unavailable. Estimates apply each request's context tier using the retained 2026-11-05 price card. Reasoning tokens are a subset of output. Response IDs deduplicate resumed exports. Missing counts remain unavailable.

### run-01: retained details

[Native sessions/settings/snapshots](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-01/state>), [all submitted sources and paired samples](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-01/submissions>), [selected source](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-01/selected-candidate.py>), [hidden checks and confirmation](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-01/confirmation.json>), [design note](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-01/workspace/DESIGN.md>).

Confirmation active: 55.31 s; confirmation GPU wait: 98.76 s.

- first correct provisional: submission 1, 247.83 active s.
- first target provisional: submission 14, 1,162.09 active s.
- selected source first measured: submission 14, 1,162.09 active s.
- Latest failure: Layout infer conflict between scores and p in T.Parallel loop:.
- [Native transcript](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-01/state/home/sessions/2026/11/05/rollout-2026-11-05T17-01-13-01a1aaf0-cf83-7903-afa8-46f8d0bdacc2.jsonl>).

### run-02: retained details

[Native sessions/settings/snapshots](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-02/state>), [all submitted sources and paired samples](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-02/submissions>), [selected source](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-02/selected-candidate.py>), [hidden checks and confirmation](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-02/confirmation.json>), [design note](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-02/workspace/DESIGN.md>).

Confirmation active: 48.24 s; confirmation GPU wait: 72.25 s.

- first correct provisional: submission 1, 244.03 active s.
- first target provisional: not recorded.
- selected source first measured: submission 14, 1,697.07 active s.
- [Native transcript](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-02/state/home/sessions/2026/11/05/rollout-2026-11-05T17-01-12-01a1aaf0-ceba-7c82-afad-0867e3ca3c4a.jsonl>).

### run-03: retained details

[Native sessions/settings/snapshots](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-03/state>), [all submitted sources and paired samples](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-03/submissions>), [selected source](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-03/selected-candidate.py>), [hidden checks and confirmation](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-03/confirmation.json>), [design note](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-03/workspace/DESIGN.md>).

Confirmation active: 65.12 s; confirmation GPU wait: 289.41 s.

- first correct provisional: submission 1, 192.64 active s.
- first target provisional: not recorded.
- selected source first measured: submission 7, 661.56 active s.
- Latest failure: Layout infer conflict between p and ph in T.Parallel loop:.
- [Native transcript](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-03/state/home/sessions/2026/11/05/rollout-2026-11-05T17-01-12-01a1aaf0-ceba-7d12-b07d-52cc115a64f9.jsonl>).

### run-04: retained details

[Native sessions/settings/snapshots](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-04/state>), [all submitted sources and paired samples](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-04/submissions>), [selected source](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-04/selected-candidate.py>), [hidden checks and confirmation](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-04/confirmation.json>), [design note](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-04/workspace/DESIGN.md>).

Confirmation active: 51.12 s; confirmation GPU wait: 31.44 s.

- first correct provisional: submission 1, 273.94 active s.
- first target provisional: submission 5, 1,203.64 active s.
- selected source first measured: submission 5, 1,203.64 active s.
- Latest failure: Layout infer conflict between scores and ps in T.Parallel loop:.
- [Native transcript](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-04/state/home/sessions/2026/11/05/rollout-2026-11-05T17-01-13-01a1aaf0-ceba-7fe2-a0b7-b7c54debc2ba.jsonl>).

### run-05: retained details

[Native sessions/settings/snapshots](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-05/state>), [all submitted sources and paired samples](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-05/submissions>), [selected source](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-05/selected-candidate.py>), [hidden checks and confirmation](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-05/confirmation.json>), [design note](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-05/workspace/DESIGN.md>).

Confirmation active: 49.67 s; confirmation GPU wait: 164.49 s.

- first correct provisional: submission 1, 184.46 active s.
- first target provisional: submission 9, 739.19 active s.
- selected source first measured: submission 9, 739.19 active s.
- Latest failure: Failed to set the allowed dynamic shared memory size to 164864.
- [Native transcript](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-05/state/home/sessions/2026/11/05/rollout-2026-11-05T17-01-12-01a1aaf0-ceba-71a2-80d6-692a78a40563.jsonl>).

### run-06: retained details

[Native sessions/settings/snapshots](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-06/state>), [all submitted sources and paired samples](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-06/submissions>), [selected source](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-06/selected-candidate.py>), [hidden checks and confirmation](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-06/confirmation.json>), [design note](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-06/workspace/DESIGN.md>).

Confirmation active: 53.37 s; confirmation GPU wait: 7.22 s.

- first correct provisional: submission 1, 246.91 active s.
- first target provisional: submission 5, 795.96 active s.
- selected source first measured: submission 15, 1,181.94 active s.
- Latest failure: Downcast from tirx.Sub to ir.IntImm failed..
- [Native transcript](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-06/state/home/sessions/2026/11/05/rollout-2026-11-05T17-01-14-01a1aaf0-cf83-7380-8836-f40138e0ea32.jsonl>).

### run-07: retained details

[Native sessions/settings/snapshots](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-07/state>), [all submitted sources and paired samples](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-07/submissions>), [selected source](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-07/selected-candidate.py>), [hidden checks and confirmation](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-07/confirmation.json>), [design note](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-07/workspace/DESIGN.md>).

Confirmation active: 52.25 s; confirmation GPU wait: 17.23 s.

- first correct provisional: submission 1, 225.14 active s.
- first target provisional: submission 11, 1,025.40 active s.
- selected source first measured: submission 11, 1,025.40 active s.
- Latest failure: Layout infer conflict between scores and ps in T.Parallel loop:.
- [Native transcript](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-07/state/home/sessions/2026/11/05/rollout-2026-11-05T17-01-12-01a1aaf0-ceba-7590-9dab-10aea88b45f0.jsonl>).

### run-08: retained details

[Native sessions/settings/snapshots](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-08/state>), [all submitted sources and paired samples](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-08/submissions>), [selected source](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-08/selected-candidate.py>), [hidden checks and confirmation](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-08/confirmation.json>), [design note](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-08/workspace/DESIGN.md>).

Confirmation active: 14.03 s; confirmation GPU wait: 6.44 s.

- first correct provisional: submission 1, 237.37 active s.
- first target provisional: submission 15, 1,421.08 active s.
- selected source first measured: submission 15, 1,421.08 active s.
- Latest failure: Pipeline planning error: Multiple writes to overlapping buffer regions detected. Stage 0 and stage 12 are both writing to buffer 'ks' with overlapping regions. This is not supported in pipeline planning..
- [Native transcript](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-08/state/home/sessions/2026/11/05/rollout-2026-11-05T17-01-12-01a1aaf0-cf83-78d1-a9a4-42398768c763.jsonl>).

### run-09: retained details

[Native sessions/settings/snapshots](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-09/state>), [all submitted sources and paired samples](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-09/submissions>), [selected source](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-09/selected-candidate.py>), [hidden checks and confirmation](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-09/confirmation.json>), [design note](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-09/workspace/DESIGN.md>).

Confirmation active: 46.93 s; confirmation GPU wait: 109.04 s.

- first correct provisional: submission 1, 219.91 active s.
- first target provisional: submission 12, 940.20 active s.
- selected source first measured: submission 12, 940.20 active s.
- Latest failure: Layout infer conflict between scores and ps in T.Parallel loop:.
- [Native transcript](</Users/dingyi/kta-data/dev/source-attention-control-v1/slots/run-09/state/home/sessions/2026/11/05/rollout-2026-11-05T17-01-12-01a1aaf0-ceba-7043-a4f8-77255c41a5f8.jsonl>).

## Matched-arm status and evidence

- prescribed: 0/3 target outcomes; 0 pending outcomes retained.
- self-selection: 1/3 target outcomes; 0 pending outcomes retained.
- no-template: 1/3 target outcomes; 0 pending outcomes retained.

No successful-run-only average is used. Development milestones are provisional. Rule reads/use and self-supplied choices remain in design notes and transcripts.

[Frozen campaign](</Users/dingyi/kta-data/dev/source-attention-control-v1/campaign.json>), [Prelaunch seal](</Users/dingyi/kta-data/dev/source-attention-control-v1/seal.json>), [Public packages](</Users/dingyi/kta-data/dev/source-attention-control-v1/packages>).

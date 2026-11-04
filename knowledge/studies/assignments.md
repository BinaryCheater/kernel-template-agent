# Parallel library assignments

This assignment overrides automatic next-library selection in the catalog. Five workers share this checkout. Only the coordinator changes ownership. Complete your ordered queue, then stop; do not take another queue. CUTLASS remains paused. Previously completed libraries are not reopened.

| Worker | Exclusive library queue | Thread |
|---|---|---|
| A | FlashInfer → FlashMLA → ThunderKittens | 01a1a2b0-af9f-72e0-93ff-574c34e2ef4e |
| B | sglang → DeepEP → TensorRT-LLM | 01a1a569-1679-7fe3-b329-339455c6ca1d |
| C | QuACK → apex → Mamba → causal-conv1d | 01a1a569-1c2c-7e43-9482-e4582b626439 |
| D | Fuser → CCCL → Triton | 01a1a569-20f4-7ba3-ab6a-b21d3ffe662d |
| E | spconv → Faiss → cuVS → cuCollections → gsplat → NCCL | 01a1a569-25c5-7683-a087-52a52c2447ba |

## Shared workspace

Use extract-kernel-templates and its representative-design reading policy. Whole-library scope does not require auditing every implementation. Source studies are not WP deliveries: no commits, GPU experiments or unrelated changes.

Write only your assigned library study directories under knowledge/studies/. Keep each library's current status in its study.md and review.md. The coordinator owns this file, the shared study README, reviewed.yaml, shared skill/procedure files and shared Template consolidation. Do not edit another worker's Templates; put proposed extensions and cross-library reuse in your own study. Read the shared seed and other studies before proposing new rules; record exact versions used. A proposal with the same identity but different content requires reconciliation, not overwriting.

Retain downloaded source in library-specific directories outside Git. Reuse existing retained sources read-only. Record new source locations in your own inputs/sources.json; do not concurrently rewrite catalog-sources.json. Preserve partial work and name unread designs when stopping. Dependencies may be read, but their full study belongs to their assigned owner.

The coordinator selected fixed queues rather than a dynamic claim service: ownership is visible here, overlap is prevented by disjoint queues, and reassignment requires editing this table and notifying affected workers. Library workloads may be uneven; queues can be rebalanced after a worker finishes. No automatic stealing.

## Required reading depth

Efficiency means skipping repetitive variant audits, not skipping distinct designs. Read a concrete representative of each materially different design family far enough to explain its mechanism, variables, constraints and applicability. Bind actual existing Template fields or extract missing knowledge with a proof outline. A directory listing, possible match or list of major unread families is partial progress, not library completion. Revisit prematurely closed assigned studies; do not reopen user-paused CUTLASS. No Template count quota. Keep validation focused on consequential uncertainties.

## Current user override

Workers A and B must checkpoint now and stop without starting or expanding another library. Record remaining work in their own study pages and their named remaining-work section in README.md (an explicit shared-file exception). Their queues remain reserved but inactive after checkpoint; no worker automatically takes them. CUTLASS remains paused.

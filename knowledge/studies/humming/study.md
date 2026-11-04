# Humming NVIDIA source study

Status: source study closed at proposed knowledge, per the user's clarified scope. All thirteen
mechanism groups have source bindings and explicit decision dispositions. Four paired rules,
source Realisations, composition bindings, replay checks and remaining gaps are delivered.
Independent admission, GPU validation and WP implementation delivery are separate work.

Assignment: extract proposed refinement Templates and versioned ProofOutlines from the full NVIDIA kernel family, including dense and expert GEMM, input preparation, weight conversion, and the compute benchmark. Follow `.agents/skills/extract-kernel-templates/SKILL.md` and the operator-library playbook. Output: this directory in the primary KTA checkout.

## Inputs

- Humming: https://github.com/vllm-project/humming at `145bed14896b34b35bd6934dc35063975b47ee28`, Apache-2.0. Complete source tarball retained outside Git under `/Users/dingyi/kta-data/dev/humming-study/inputs/`.
- vLLM callers: catalog revision `c1d8fade8d392a010b80aaa0de55f6f0f02a4ebb`; mixed-precision linear and fused Humming MoE adapters, plus four helpers followed to resolve the linear conversion/application boundary. This partial archive omits other vLLM modules.
- Starting pool: repository seed Templates, reference candidates, ProofOutlines, Realisations and Patterns, fixed by file digests in [inputs/pool-snapshot.json](inputs/pool-snapshot.json). The repository navigation and empty reviewed-study index are also pinned; no sibling-study
proposal is a dependency. This snapshot does not claim to include a live knowledge store.
- Source provenance and integrity: [inputs/sources.json](inputs/sources.json).

## Scope and stopping condition

Inventory exports, JIT builders, native launcher registrations, dispatch/target gates and every device body. Trace a concrete NVIDIA witness for each mechanism group; account for shared specializations and neighboring variants. Keep unsupported or unverified boundaries explicit. Match the pinned pool before introducing a rule. Pair each new or revised rule with its ProofOutline; validate schemas and replay distinguishing source bindings. Perform one correction pass and close each decision as explained, partial or open with its next evidence source. Deliver `inventory.md`, `trace.md`, `delta/`, `checks/` and `review.md`. The pass ends at proposed knowledge; admission is separate.

Source analysis and small index/numeric witnesses are the evidence domain. GPU compilation, timing, emitted SASS and full-kernel correctness are separate claims requiring remote execution evidence. No GPU work has run in this study.

## Workspace

The checkout is on `wp/PERFMODEL`. Performance-model work present at assignment start and
untracked navigation/runbook/skill files are unrelated; preserve them. Authored study output is confined to `knowledge/studies/humming/`. No runtime or schema change was made. This is a repository source study rather than a
new WP implementation delivery; no old WP delivery is overwritten.

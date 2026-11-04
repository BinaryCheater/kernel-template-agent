# Marlin NVIDIA source study

Assignment: continue the NVIDIA catalog after Humming and DeepGEMM. Primary output is knowledge/studies/marlin/. One library at a time.

Pinned source: IST-DASLab/marlin at the catalog revision, Apache-2.0; complete 18-file archive and tree retained outside Git. [Manifest](inputs/sources.json) records absolute paths and hashes. All file blobs match [the retained Git tree](inputs/git-tree.txt). Keep the source and archive.

Starting pool: [pinned file snapshot](inputs/pool-snapshot.json), seed/reference baseline plus provisional Humming and DeepGEMM candidates. Reviewed index remains empty. Reuse by name/version and pinned body; no proposal has been independently admitted.

Scope: all NVIDIA entry/build/device paths in this repository revision: Python mul/Layer packing, C++ binding, CUDA template/dispatch, striped partial reduction, tests and GPTQ/example callers. Related vLLM material in the source cache is reference-only unless needed to explain this pinned family.

Stopping condition: every inventory group has a source-backed disposition; choices mapped to existing rules or explicit gaps; new candidates paired with ProofOutlines and discriminating checks; one correction pass and short review. Source extraction ends at proposals. Admission, GPU execution, WP gates and commits are separate assignments.

Status: source study complete. All 18 retained files, one extension entry, one CUDA family and ten compiled specializations have source-backed dispositions in [inventory.md](inventory.md) and [trace.md](trace.md). Five existing mechanisms are reused conditionally; no new Template is proposed. Four source-equation/schema replays and two lint/format commands pass. [Review](review.md) is the reading entry point. Independent admission, compiled-device behavior and GPU synchronization/numerical checks remain separate work. No CUDA compile or execution occurred.

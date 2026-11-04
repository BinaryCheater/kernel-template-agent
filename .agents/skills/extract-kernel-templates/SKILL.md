---
name: extract-kernel-templates
description: Extract or review KTA refinement Templates and proof outlines from a kernel or operator family. Use for source studies, pool deltas, and reconciling competing decompositions.
---

Use the [accumulation guide](../../../docs/runbooks/learndemo-1/template-accumulation.md) for naming, granularity, the three deliverables, and a worked example. Write candidate records in the existing [schema](../../../spec/schemas.md) when delivering a pool delta. In a comparative review, reference existing candidates by name and version.

**Read the shared pool first.** Use the [knowledge library entry point](../../../knowledge/README.md) to discover seed rules, reviewed study additions, and pending proposals. Pin the versions actually available to the study. Reference each binding by name and version. Show a chain when a dependency between rules needs explaining or checking.

**Open the assigned workspace.** Follow the [operating procedure](../../../docs/runbooks/learndemo-1/operator-library-playbook.md) for directory selection, filenames, execution order and closure. Record the current assignment, pinned inputs and stopping condition before tracing. Resume from existing artifacts when continuing a study.

**Inventory the assigned scope.** A library assignment covers its full NVIDIA kernel family. Inventory entry points, build registrations, dispatch branches and device bodies. Group specializations that share a mechanism, trace a representative of each group and account for its differences. Reconcile the final inventory against source. A single-entry study applies when explicitly requested.

**Extract designs, not an implementation audit.** Survey the full library by distinct computation organization, layout, pipeline, quantization and scheduling. Deep-read one representative of a distinct design; inspect related implementations only for differences that add a reusable choice, constraint or applicability case. Record concrete reuse briefly and move on when existing Templates explain a variant. Add checks only to resolve a consequential Template uncertainty. Note unrelated bugs/host edge cases without pursuing them. Coverage closes when distinct designs have dispositions and remaining gaps are named, not when every entry has its own trace or test. Honor any user time limit across turns; stop the library at its fixed deadline without extending for cleanup or validation.

**Stay on the NVIDIA path.** Record the target architecture and selected backend. Follow the operating procedure's language-specific route from CuTe DSL, TileLang, Triton, CUDA or PTX to one concrete witness. Keep target gates, JIT menus and measured performance separate.

**Account for each design choice.** Keep one row per observed decision: source location and concrete binding | rule and field (or gap) | evidence or open question. Distinguish independent variables, derived quantities, legal constraints, compiled menus and dispatch heuristics. Include workload and target scope. Cite evidence for structural dependencies. Record inferred freedoms and performance couplings as hypotheses with their scope.

**Recover the instance.** Substitute the reference bindings, follow the chosen branch, and check that the rule predicts the relevant indices, values, or resource events. For a layout, test identifiable coordinates; repeated low-bit values can hide a swap. For a numerical transformation, examine grouping, rounding and range boundaries. Pair every new or changed rule with its versioned ProofOutline. State which premise each check covers and what remains open.

**Compare or reorganize by meaning.** Two decompositions are equivalent for the reviewed scope when their bindings express the same choices, constraints, effects, interfaces and obligations in both directions. A choice mentioned only in prose remains a gap if a solver needs it. For a split or merge, map each old field and obligation to its new owner, keep the old version and source instances addressable, and replay affected bindings. Record any deliberate domain change and reopen proofs that used a changed premise.

**Write for the next kernel.** Open the summary with a structural retrieval cue: given this workload and current design, what observable situation makes the choice worth considering? Then state the invariant and the supported binding or implementation boundary. Preserve executable scope guards. A new use with the same mapping and premises adds an instance; a new tail path, mapping or premise requires a variant or revision. For the hardest decision marked reused, point to the old field or Realisation that supplies it; capture a missing decision as a delta or gap. Distinguish choices fixed by the workload from choices the optimizer can vary. A reader should be able to instantiate the rule without opening the source study.

**Continue a catalog assignment.** Use the study index and source-retention rules in the operating procedure. Keep downloaded sources and their absolute locations. Close each library once, then advance to the next outstanding library under the same goal.

**Finish.** Use the existing review page as a short menu: kernel need → linked rule → useful choice. Give one composition and link the evidence once. Source studies finish when the extraction and relevant checks are recorded. A WP gate, commit, formal promotion or further GPU work is a separate assignment. Close the goal when its assigned scope is exhausted.

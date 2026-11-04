# triton source study

Status: representative-design extraction complete; pending independent review.

Worker D exclusive queue: Fuser → CCCL → Triton. This library was resumed after the initial pass closed before distinct families were read. The resumed traces and reconciled inventory supersede the early unread-family dispositions.

Source: https://github.com/triton-lang/triton, revision `3412130d638105963a34c3e8c622a98f843625ae`. [Source manifest](inputs/sources.json) retains the absolute full-superproject tree/archive paths and omitted submodule scope. Sources remain outside Git. License: MIT (retained LICENSE).

Pool: [exact file/identity pins](inputs/pool.json), all eight seed rules and the provisional rules named by the traces. Proposals remain local to this study; shared ownership/index/admission files are unchanged.

Boundary: NVIDIA source designs by computation, layout, pipeline, quantization and scheduling. Read one representative and the meaningful differences; no implementation audit, compiled artifact, performance measurement, Store admission or commit.

Outcome: 23 candidate/outline pairs plus concrete existing-rule reuse. [Reconciled inventory](inventory.md), [representative traces](trace-design-families.md), [earlier bindings](trace.md), [review menu](review.md) and [checks](checks/results.md). Each remaining native visibility/lifetime/numerical obligation is distinguished from unread source-design work.

Stopping condition: after the assigned three-library queue. Worker D queue ends here; no other queue is claimed.

Final audit includes the shipped production package; [production traces](trace-production-kernels.md) account for every materially distinct NVIDIA production family alongside the tutorial surface.

Closure audit: public NVIDIA entry-point families, selected dispatch differences and concrete device bodies reconcile with inventory.md. Fuser includes scheduler/runtime and opt-in integration boundaries; CCCL includes CUB, Thrust, cudax and ISA wrappers; Triton includes tutorials, Gluon and shipped production kernels. Each distinct design has a concrete binding or paired local proposal. Dependency internals assigned to other workers were not revisited. checks/validation.md records successful schema, identity, link, retained-source and semantic-witness checks.

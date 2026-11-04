# cccl source study

Status: representative-design extraction complete; pending independent review.

Worker D exclusive queue: Fuser → CCCL → Triton. This library was resumed after the initial pass closed before distinct families were read. The resumed traces and reconciled inventory supersede the early unread-family dispositions.

Source: https://github.com/NVIDIA/cccl, revision `b31ec15aa3424670e0878ed770ef80d51b36fa0b`. [Source manifest](inputs/sources.json) retains the absolute full-superproject tree/archive paths and omitted submodule scope. Sources remain outside Git. License: Component licenses retained; Apache-2.0 with LLVM exceptions and BSD component headers.

Pool: [exact file/identity pins](inputs/pool.json), all eight seed rules and the provisional rules named by the traces. Proposals remain local to this study; shared ownership/index/admission files are unchanged.

Boundary: NVIDIA source designs by computation, layout, pipeline, quantization and scheduling. Read one representative and the meaningful differences; no implementation audit, compiled artifact, performance measurement, Store admission or commit.

Outcome: 14 candidate/outline pairs plus concrete existing-rule reuse. [Reconciled inventory](inventory.md), [representative traces](trace-design-families.md), [earlier bindings](trace.md), [review menu](review.md) and [checks](checks/results.md). Each remaining native visibility/lifetime/numerical obligation is distinguished from unread source-design work.

Stopping condition: after the assigned three-library queue. Continue with the next assigned library.

The final audit includes cudax cooperative groups/reductions and driver copy/fill paths; their source-backed dispositions are in the inventory and design trace. cuCollections dependency internals remain Worker E ownership.

Closure audit: public NVIDIA entry-point families, selected dispatch differences and concrete device bodies reconcile with inventory.md. Fuser includes scheduler/runtime and opt-in integration boundaries; CCCL includes CUB, Thrust, cudax and ISA wrappers; Triton includes tutorials, Gluon and shipped production kernels. Each distinct design has a concrete binding or paired local proposal. Dependency internals assigned to other workers were not revisited. checks/validation.md records successful schema, identity, link, retained-source and semantic-witness checks.

# Operating procedure for a kernel source study

Use this procedure for the assigned library, operator, or comparison. The [field guide](template-accumulation.md) defines the knowledge records. The [source catalog](operator-source-catalog.md) helps locate material when a question calls for it.

## Workspace and files

For repository work, use the primary KTA checkout. Save authored study records under `knowledge/studies/<library>/`; resume the pinned study there and use a revision subdirectory for a new snapshot. Keep downloaded source, large artifacts and session logs under the configured KTA data root and reference them by revision or record ID. Use a temporary directory only when requested. Record the destination in the handoff.

| Location in a manual study | Contents |
|---|---|
| `study.md` | The question, operator boundary, stopping condition, source URL/revision/license, input pool snapshot or version list, and current status. Record relevant omissions in partial archives. |
| `inputs/` | Read-only pinned source or a manifest pointing to an existing source archive or Report, plus the pool snapshot reference. Store supporting papers and external examples here with provenance. |
| `trace.md` | Caller contract, selected dispatch path, branch conditions, concrete bindings, and an ordered decision table. Each row cites a revision-relative source path and lines and links to a candidate field, existing rule or gap. |
| `delta/` | A short index of proposed changes. For each new or revised Template use `delta/<name>@<version>/template.yaml` and `proof.yaml` in the KTA schemas. Record source-to-version bindings. Use `delta/chain.yaml` when a concrete composition clarifies or tests a cross-rule dependency. Identify partial chains explicitly in the index. Record split/merge translations here. An empty delta still records which versions were reused. |
| `checks/` | Small replay scripts, inputs and observed outputs, or references to stored Check, ProofResult or measurement records. Include the command, tested domain and outcome. Keep raw artifacts here or in the object store. |
| `review.md` | One worked example, outcome per in-scope decision, checks with evidence links, unresolved obligations and proposed admission status. This is the reading entry point. |

Keep independent model runs in sibling study directories with the same pinned inputs. On resumption, read `study.md`, verify the revision and snapshot, then resume at the first missing output. Keep source checkouts, logs and generated artifacts outside the KTA Git tree. A study directory contains proposals; the knowledge store is updated through KTA admission as a separate operation.

## Continuing across libraries

Use one worker and one library at a time. For a catalog assignment, take the next outstanding library from the study index, proceeding through the linked catalog in reading order. Count each distinct repository once. Treat links used only to explain a caller, instruction, or test as references for that case. Group multiple catalog entries for the same library in one inventory. When the assigned catalog is exhausted, close the goal. A missing source blocks that library; record the reason once and continue with the next available library.

Maintain one row per started library in [the study index](../../../knowledge/studies/README.md): library, pinned revision, status, source manifest, and result page. Resume an unfinished row before opening a fresh one. Write each rule once; reuse its pinned identity in other kernels. After the library inventory has dispositions and the candidates have usable summaries, outlines and relevant checks, publish its short review and advance. Leave independent admission to the reviewer. Reuse shared checks. Add a script only for a concrete failure mode; record timing and usage from the session when available. Avoid further polishing after one correction pass.

**Keep sources.** Before downloading, consult earlier manifests and the local cache inventory. Reuse an exact revision. Keep new source outside Git under the configured KTA data root at `dev/operator-sources/<owner>/<repo>/<revision>/` or retain an existing persistent snapshot. Store the URL, revision, absolute local tree/archive paths and full/partial scope in the study's `inputs/sources.json`. Keep any source archives, submodule snapshots and generated inputs that were actually used, and identify missing pieces. A partial Git mirror can supply a tree listing; materialize and retain the actual blobs needed for the study. Verify retained material can be opened before closing. Temporary analysis and context compaction must never delete source inputs.

## Library scope and acceptance

Read by distinct reusable designs, not by implementation count. Survey computation organization, layouts, pipelines, quantization and scheduling; deep-read a representative and inspect relatives only for new choices, constraints or applicability. Brief concrete reuse is enough for an explained variant. Write/run a check only for a consequential Template uncertainty; note unrelated host bugs without pursuing them. Library closure accounts for distinct designs and names remaining gaps; it does not require a detailed trace/custom check per entry. For a user time limit, persist the fixed wall-clock deadline and advance at it with a partial-study status and unexamined families; validation or reporting does not extend the limit.

For a library assignment, create `inventory.md` before extraction. Cross-check public operations and build/dispatch registrations against device source files. Group entries by distinct computation, layout, schedule or execution mechanism. Map shared implementations and generated variants back to a group. Trace each group, reuse existing rules, and record a source-backed disposition for each branch. Flag inaccessible paths as gaps with their coverage impact. A representative trace closes its group only when the other members and differences are accounted for.

The executor hands over records, replayable checks, outstanding gaps and available session timing/usage. An independent reviewer checks the inventory against the pinned source, replays selected bindings and judges quality and efficiency. Self-reported coverage is input to that review.

## Execution order

1. **Scope.** Write `study.md` and resolve inputs. If an essential source is unavailable, record the missing path and close the affected scope. Follow additional files only when they answer the recorded question.
2. **Trace.** Map the caller, visible outputs and side effects. Follow preparation, build specialization, runtime selection and device execution where applicable. For each mechanism group choose a concrete input; add a neighboring variant when it distinguishes competing explanations. In `trace.md` record the actual branch, derived values, consumed/produced objects and the last verified boundary.
3. **Match and compose.** Query the pinned pool and bind its versions before proposing changes. Use a local binding for a standalone choice. Show a short chain when its interface matters; a whole-kernel reconstruction is optional for extraction. Give every scoped decision a disposition, including unresolved behavior.
4. **Write the delta.** Apply the update table below. Pair every new or revised rule with a proof outline and explicit evidence for its scope. For an outstanding proof obligation identify the missing premise and the check that could resolve it.
5. **Verify.** Validate candidate records against the current schema, check references and substitute source bindings. Test a witness that can distinguish the proposed explanation from a plausible wrong one. Preserve failures and distinguish checks actually run from planned checks.
6. **Close.** Write `review.md`. After one correction pass, mark each decision explained, partial, or open and set the final study status. Lead the review with a short menu of usable rules and one concrete application. Hand back the study directory and review link. Close an extraction goal here; WP delivery and commits are separate assignments. Record unrelated questions for later selection; the assignment ends here.

## Read across implementation languages

Scope each assignment to an NVIDIA GPU path. Record compute capability, input/output and accumulator types, tensor layouts, shape regime and the branch actually selected. Host code, generators and tests supply the evidence for that path.

| Source form | Reading order | Evidence to keep |
|---|---|---|
| CuTe DSL | Python entry and dispatch → tiled copy/MMA layouts → architecture atom and pipeline | Follow one logical coordinate into a lane or fragment. Record which layout is declared and where the backend mapping becomes unknown. |
| TileLang or Triton | Shape signature and candidate menu → parallel loops/program IDs and predicates → selected NVIDIA lowering | Distinguish tensor shape, tunable tile, derived extent and compiled specialization. Use generated code only when available for the exact configuration. |
| CUDA C++ | Launcher and template instantiation → CTA/warp/lane ownership → shared-memory stages, barriers and epilogue | Follow one full tile and one tail; record which thread writes each visible result and when it is safe to reuse storage. |
| PTX or inline assembly | Caller predicate and target → operand types/layout/address spaces → instruction sequence and completion protocol | Use the [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) for instruction semantics. A wrapper shows available instructions; a selected caller or emitted artifact establishes use. SASS and timings require their own evidence. |

## Compose a chain

Start with the workload contract and instantiate rules against a specific operation or intermediate. For each step record the Template version, variable bindings, produced object and consumers. Connect steps with producer/consumer edges; add ordering edges where values, ownership, storage lifetimes or synchronization require them. Put constraints that depend on this combination in the chain/family binding or glue. Record them as a general Template relation only when the relationship holds in the stated scope. A performance-only interaction belongs in a scoped Coupling with its evidence.

For example, in W4A16 Marlin, the packer establishes the layout of a weight fragment. The decoder consumes that layout and establishes its numerical values. Scale lookup supplies the scale for the corresponding group. The reduction can apply a shared scale after a group sum when the scale is constant over that sum and the numerical contract allows reassociation. Check each edge with a witness: a unique coordinate for layout, all 16 codes for decode, two groups with unequal scales for scale ownership. Check the combined chain as well: a correct packer and a correct decoder may use different layout conventions.

## Update decisions

| New observation | Proposed pool change | Required review |
|---|---|---|
| Same contract and obligations, different parameter values | Bind the existing version in a new instance | Substitute the new values and check constraints. |
| Same transformation, different ISA or backend code | Add a Realisation and scoped capability requirements | Check the backend implements the same mapping and numeric contract. |
| A new shape or device appears to benefit | Attach a scoped Finding with its evidence basis | Keep measured regimes, inferred benefit and legality distinct. |
| A valid reference falls outside a declared domain, or a counterexample invalidates an old premise | Propose a version with the corrected scope, constraints or obligation | Record the witness and whether the domain widens or narrows. Revisit affected chains and proofs. |
| Choices can vary independently and each has its own object and obligation | Split the rule and specify the intermediate contract | Map every old binding to the new chain. Test old examples plus a witness exercising the new independence. |
| Two records express the same transformation | Propose a canonical rule with mappings from both originals | Check both directions for choices, constraints, resource effects, numeric contract and obligations. Keep separate versions if equivalence is only partial. |
| A reference requires a new design object and local obligation | Propose a new Template and ProofOutline | Compare the closest old rule and show the concrete choice it cannot express. |
| Evidence does not settle the abstraction | Keep a Gap or candidate hypothesis | Name the competing interpretations and a source or counterexample that separates them. |

For every split, merge or replacement, keep old versions and historical bindings addressable. Deliver a migration table with old version/field → new version/field, translation, and affected evidence. Replay old bindings on the old scope; mark intentional differences and which proofs need reopening. Propose the delta against the snapshot read. If the live pool changed meanwhile, recheck duplicates, dependencies and version conflicts before admission. A source study ends at proposed knowledge. Promotion to measured or proved needs the corresponding evidence.

A reviewer should be able to answer three questions from the delta alone: What choice can the next Family express? Which invalid choice will the constraints reject? Which claim still needs evidence?

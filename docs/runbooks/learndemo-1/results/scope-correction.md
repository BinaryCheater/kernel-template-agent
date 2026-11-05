# Correction: source-kernel reconstruction is not yet established

On 2026-11-05, the user pointed out that execution had departed from the earlier agreed experiment. This was an existing requirement, not a newly added scope: select a specific studied production kernel, supply the relevant extracted Templates to an independent implementation session, and compare its reconstruction directly with that original kernel. A second mode must find the relevant Templates in the full library without a prescribed subset.

The earlier planning discussion explicitly said to select the same production kernel for both modes, define reproduction as the same computation contract with competitive performance, measure the reference on the same GPU, and provide the reference implementation's location. It also explicitly allowed different generated code, layouts and schedules. The handoff plan weakened this into two representative computations and an unnamed production reference. Execution then chose PyTorch entry points without establishing that they ran the studied reference kernel. The fault therefore began in the handoff and continued through implementation, acceptance and reporting. The retained evidence is `original-agreement-audit-2026-11-05.json` under the experiment data root.

The completed campaign tested generic GEMM and causal-attention implementation from the consolidated library. It did not freeze a studied source kernel as the reconstruction target or benchmark that exact source kernel. Its completion claim for source-kernel reconstruction is withdrawn. Numerical results, native sessions, accounting and original snapshots remain valid records of the narrower experiments and are retained unchanged.

The actual attention performance reference in `tools/kta_template_replay/worker.py::production` is PyTorch `scaled_dot_product_attention`; GEMM uses `torch.matmul`. A later [backend trace](attention-backend-audit.md) in the original pinned environment identifies PyTorch's integrated fused FlashAttention kernel on all three original attention shapes. It was not an explicitly forced unfused math baseline. The campaign still did not identify and run the pinned studied FlashAttention/FlashInfer source specialization. These numbers cannot establish recovery of that source kernel. The FP64 mathematical computation is a separate correctness oracle.

The missing evidence is:

- An exact source-library revision, kernel entry/specialization and original runtime invocation, with its shape/layout/dtype/mask/output contract.
- A trace from that kernel's important design decisions to the supplied Template identities/bindings, including gaps and decisions filled in by the implementing model.
- A real run of that original kernel and the independently reconstructed kernel on the same hardware, inputs and timing boundary.
- Separate findings for numerical agreement, Template use and model-supplied choices, performance against the original kernel, and implementation time/token/cost. Code/layout/schedule identity is not required by the original agreement. A faster implementation against an unrelated comparator does not establish the requested result.

Existing starting evidence includes `knowledge/seed/references/flash_attention_2_fwd/{source.yaml,decomposition.md,chain.yaml,reconstruction.yaml}` and the pinned FlashAttention and FlashInfer study traces. Those records must be reconciled to the exact chosen source specialization before another implementation session starts; source-study work need not be restarted. The source kernel stays evaluator-side. The implementer receives the computation contract, Templates and language tools, without the original implementation or an answer chain.

The next experiment must establish this source-to-Template-to-reconstruction chain before the proposed unseen-shape or W4A16 extensions. No source-reconstruction model run has been launched by this correction.

The process failure was that the coordinator changed the question into generic implementation, judged completion against its own executable checks, and reported correctness/performance figures without naming the missing source-kernel link. Subsequent explanations shortened the same misplaced conclusion instead of rechecking the experimental claim.

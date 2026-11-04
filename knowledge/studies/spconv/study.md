# spconv NVIDIA source study

Status: extracted by distinct retained NVIDIA design; Worker E, first library. Stop after the assigned six-library queue. No commits, GPU experiments or admission.

Scope: all NVIDIA computation designs in the pinned superproject, grouped by indexing, sparse contraction, pooling and voxelization. Generated CUDA is read through its Python generator; external cumm implementation is a named boundary.

Inputs: [retained source](inputs/sources.json). Seed baseline: output_tiling@2, reduction_partition@1, physical_storage@1, compute_instruction_selection@2, async_transfer@1, multi_buffer_rotation@2, shared_resource_binding@1, shared_memory_tiling@1. Reviewed index: templates=[] at study start. Other study candidates remain provisional.

Stopping condition: inventory dispositions and reusable choices recorded, with unread designs named. No implementation audit.

Progress: distinct retained NVIDIA design groups now concretely traced in trace-additional-designs.md. Candidate validation and final local reference/identity reconciliation passed.

Source-study closure: the concrete continuation supersedes initial unread dispositions. Local candidates validate, retained sources reopen, and their identities do not conflict with other study candidates at reconciliation. Device realization/proof and explicitly identified external dependency boundaries remain open; no major retained local design is closed from a directory listing alone.

Worker E queue closure: all six assigned libraries have representative-design dispositions. Final schema, retained-source, local-link and candidate-identity checks passed; see checks/queue-reconciliation.json. Stop after NCCL. No commits or GPU experiments were performed.

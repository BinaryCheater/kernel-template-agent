# Faiss reusable choices

Status: extracted by distinct retained NVIDIA design; pending independent review.

Streaming top-k with distributed arrivals -> [collective-triggered queues](delta/collective_triggered_candidate_queue@1/template.yaml) -> choose lane staging and retained capacity, preserving full-warp votes. Dense distance tiles -> seed output_tiling@2. Query-constant L2 norm -> defer until after rank selection only under the chosen numerical contract.

Worked case: k=33 uses capacity 64 and local queue 3. One lane filling triggers the entire warp merge; the final flush includes every remaining lane queue. Equal distance IDs are unspecified under default Comparator, so no smallest-ID guarantee is added.

[Trace](trace.md) and [inventory](inventory.md) record source substitution and boundaries. No executable replay or GPU run performed; full-warp tail participation and sentinels remain device realization obligations. Unread designs: exact IVF interleaving/quantizer codecs, binary tails, append ownership and ICM encoding. This pass proposes the queue invariant, not complete coverage of those families.

## Distinct-design continuation

[Additional trace](trace-additional-designs.md) now provides concrete readings for IVF/SQ/PQ layouts, binary distance, append ownership and ICM. Use [interleaved append](delta/warp_interleaved_code_append@1/template.yaml) for same-feature warp loads; [lookup distance](delta/lookup_factorized_code_distance@1/template.yaml) for codebook-factorized metrics; [midpoint quantization](delta/truncated_range_midpoint_quantization@1/template.yaml) for the exact SQ format; [conditional minimization](delta/conditional_code_coordinate_minimization@1/template.yaml) for coupled additive codes.

Replay: `.venv/bin/python knowledge/studies/faiss/checks/replay.py`; [observed result](checks/replay-result.json). All256 nibble pairs preserve the old prefix and new field; an overwrite-by-OR counterexample distinguishes append from replacement. Midpoint witnesses distinguish truncation from nearest-even. These cover the local bit/format premises; CUDA shuffle lowering and exact FP32 boundary behavior remain realization obligations. Five Template/ProofOutline pairs pass the current Pydantic schemas.

The initial paragraph naming these families unread is superseded by this continuation. Remaining obligations concern specific realizations (native shuffles, FP32 boundary rounding, device memory scope), not unvisited major design families. External cuVS implementations are addressed in their own study; no CPU algorithms included.

Source-study closure: the concrete continuation supersedes initial unread dispositions. Local candidates validate, retained sources reopen, and their identities do not conflict with other study candidates at reconciliation. Device realization/proof and explicitly identified external dependency boundaries remain open; no major retained local design is closed from a directory listing alone.

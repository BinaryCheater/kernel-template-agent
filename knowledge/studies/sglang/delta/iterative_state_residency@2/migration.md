# Migration from iterative_state_residency@1

The old record is unchanged and remains addressable. Bind every old state_site and p field identically and set output_retention=final_only. The new output condition is then vacuous; old capacity, local dependencies, fixed numeric/termination, synchronization, lifetime and final failure/publication obligations retain their owners unchanged. All old final-only program steps remain.

Version2 deliberately adds a domain: bounded finite-state walks and per-step output tapes. output_retention=per_step adds exact output writes and generation visibility; its new step_outputs obligation owns this evidence. Existing step/lifetime/completion obligations remain. Immutable input streams do not count as simultaneously live mutable state, but current-step workspace still does. Old matrix-solve instances do not prove walk-specific row selection, random identity, NaN or CDF premises. No old source is relabeled as version2 without this binding map.

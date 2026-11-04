# Migration from keyed_weighted_race_selection@1

Keep every old field and obligation, set score_form=linear_log_uniform,uniform_map=midpoint and retain split_width8192. The old support and log/divide/tie premises imply the corresponding version2 branch; original TextSeal key/context/mixture, support and endpoint meanings are unchanged. split_width additionally admits1024.

Version2 adds finite log-weight input and a monotone Gumbel score. Its native score-recipe admission replaces the old log/divide wording but retains that old recipe as the unchanged linear branch. Changing score_form does not silently reverse uniforms: complement_midpoint is a separate boundary binding. Exact arithmetic order equivalence reopens numerical rounding/tie obligations for every new recipe. Per-token Philox quad/word and vocabulary-shard alignment are additional realization evidence, not a new model-independent RNG guarantee.

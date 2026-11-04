# Migration from operator_fusion@3

All old fields, fused_ops values, locality/checkpoint/dense/pair premises and obligations remain. New pre_plain_pair is vacuous for every old binding; old interleaved_swiglu keeps its exact clipped/shifted linear+1 recipe.

The added adjacent_silu_mul branch owns ordinary SiLU(gate)*up, producer cast-before-activation and paired coordinates. same_values/no_cross_tile_dependency retain the pairing/locality proof; existing admitted numerical semantics additionally cover exp/divide/FTZ and final conversion. A source with pairs split across independent tiles cannot bind the new branch. No old source is silently reinterpreted as ordinary SwiGLU.

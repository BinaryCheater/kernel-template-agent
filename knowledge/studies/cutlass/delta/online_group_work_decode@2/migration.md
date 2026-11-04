# Version1 projection and scalar variant

Version1 is unchanged. Every old variable maps identically; add
`search=warp_prefix_32`. Its constraints, scope, effects and32-lane program remain.
The new `scalar_forward` branch requires `cache_shapes=false`; it retains no shape
windows. Old counts/immutability/monotone-query/Int32 obligations remain. Spatial
decode stays external in both versions. No old Realisation is relabeled as scalar.

The MoE instance derives counts lazily from endpoint metadata rather than loading
a count array. Its count construction is family glue and its source requires at
least one expert. Scalar method checks replay the exact start/end cursor; old
warp-window bindings and source checks remain addressable at version1.

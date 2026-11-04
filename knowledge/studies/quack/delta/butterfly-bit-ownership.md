# Relation to butterfly_linear_transform@1

The existing H<=512 candidate couples a normalized block Hadamard and specific thread grouping. Keep it addressable. New [butterfly_bit_ownership@1](butterfly_bit_ownership@1/template.yaml) introduces stage bit ownership and inverse final store, supports source H<=32768 and preserves a caller-defined scale/pad/crop workload. This is not a silent version/domain widening.

Old H→transform_extent only for power-of-two normalized blocks; V→elements_per_thread only if its source fragment meaning agrees; threads is derived H/EPT times row packing, not a free translation. Old normalization becomes the workload scale. New stage_bit_sets and final ownership map need separate bindings. Proof outline beside YAML owns stage coverage, bijective exchange and final-store identity; source plan consumed_bits assertion at hadamard.py:487–494 checks each logical bit occurs once.

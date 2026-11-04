# Migration from power_of_two_scale_encoding@1

Keep site and every original metadata coordinate. Old normal exponent-126..127,p.scales_positive_normal_pow2 implies the version2 positive-representable-power premise. Old padding_not_logical_scale implies padding_not_substituted_for_live_scale. The old value_identity,metadata_identity,padding_exclusion obligations retain their ownership; live scales still decode exactly.

Version2 admits exactly one extra FP32 subnormal power2^-127, stored as live code0, and refines padding_exclusion to prohibit substituting padding for any required live coordinate rather than categorically forbidding all live0. Old padding0 remains padding. Zero,infinity,reserved255 and arbitrary subnormals stay excluded. Reopen the value/domain proof only for this extra minimum case; native raw-scale rounding is owned by the quantizer, not this encoding rule.

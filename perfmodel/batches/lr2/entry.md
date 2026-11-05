# Batch lr2: entry block

Round two of the funnel on `lr`'s shapes. Constants `tools/kta_bridge/constants-v4.json`: `lr`'s measured rows added to the calibration, hardware constants unchanged, MoE's register need raised by the dequantisation's temporaries. Measured assignments are cut out of the MIP; the shortlist is compiled and the top 8 after compile records go to the GPU (`perfmodel/testset/search/lr2.json`).

Question: does measuring a few and refitting repair the MoE shortlist, which lost to its anchor in `lr`? Refuted if MoE's round-two best stays above the anchor's 786.8 µs. `lr` is calibration here, so round two's rows are the first to test v4.

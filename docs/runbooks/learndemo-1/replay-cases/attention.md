# attention

O = softmax(Q @ K.transpose(-2,-1) / sqrt(D) + causal_mask) @ V. Q,K,V contiguous [B,H,S,D]; mask excludes key positions greater than query. Equal head counts, no dropout, backward or LSE output.

Shapes: [[1, 8, 512, 64], [1, 8, 2048, 128], [1, 8, 513, 64]]. All inputs/output FP16, accumulation/statistics FP32. Return one new contiguous CUDA FP16 tensor from candidate.py::entry(q, k, v). Inputs must remain unchanged. No persistent output caching, hidden host compute, shape removal or semantic changes. Include all packing/preprocessing/dispatch per call.

Target: NVIDIA GB10 sm_121; language: TileLang 0.1.15. Tolerance: atol=0.01, rtol=0.01; finite outputs required. Budget: 3600 active seconds, 20 evaluation batches, 3 shapes per batch. Timing: 10 warmups, 50 event samples, warm-input-reuse; performance aim 0.9.

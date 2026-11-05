# gemm

C = A @ B; A[M,K] and B[K,N] row-major, no bias or activation.

Shapes: [[1024, 1024, 1024], [32, 4096, 4096], [127, 1000, 769]]. All inputs/output FP16, accumulation/statistics FP32. Return one new contiguous CUDA FP16 tensor from candidate.py::entry(a, b). Inputs must remain unchanged. No persistent output caching, hidden host compute, shape removal or semantic changes. Include all packing/preprocessing/dispatch per call.

Target: NVIDIA GB10 sm_121; language: TileLang 0.1.15. Tolerance: atol=0.01, rtol=0.01; finite outputs required. Budget: 3600 active seconds, 20 evaluation batches, 3 shapes per batch. Timing: 10 warmups, 50 event samples, warm-input-reuse; performance aim 0.9.

# Matrix product with four-bit quantized weights

Implement one matrix product for inference on a quantized linear layer.

C = A @ dequant(B), where dequant(q)[k, n] = (q[k, n] - 8) * s[k // g, n]

- A: FP16 activations of shape [M, K], row-major. M is small and varies per call,
  typically 1 to 64 and sometimes larger.
- B: symmetric four-bit integer weights of shape [K, N]. Weights are constant after model
  load. The stored layout of B and s may be chosen once at load time.
- s: FP16 scales, either one row per output column or one row per group of g = 128 rows of K.
- C: FP16 output of shape [M, N], row-major.
- K and N are in the thousands. K is a multiple of 128 and N a multiple of 256.
- A caller-owned workspace may be supplied, one per concurrent call.

Target: NVIDIA GB10, compute capability 12.1, CUDA C++ with inline PTX allowed.
Objective: lowest GPU time per call for every M, with FP16 output accuracy comparable
to a float32 reference computed from the unpacked weights.

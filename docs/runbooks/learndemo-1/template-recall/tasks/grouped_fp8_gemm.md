# Grouped FP8 matrix product with masked rows

Implement one grouped matrix product used for expert routing.

For each group g and valid row m < length[g]:
D[g, m, n] = sum over k of A[g, m, k] * sa(g, m, k) * B[g, n, k] * sb(g, n, k)

- A: FP8 E4M3 [G, M, K], row-major. B: FP8 E4M3 [G, N, K], K-major.
- sa and sb are FP32 multiplicative scales that are powers of two, constant over blocks of
  128 along K. sa is per row and sb is per 128 columns of N.
- D: BF16 [G, M, N], row-major. M is the row capacity. length is an int32 device
  vector, so the valid row count of each group is unknown to the host when launching.
- Many groups have few or zero valid rows.
- Representative sizes: G up to 512, M = 128, N up to 4096, K up to 4096.

Target: NVIDIA compute capability 9.0, CUDA C++ with inline PTX allowed, asynchronous tensor
memory access engine available.
Objective: lowest GPU time per call, with output accuracy comparable to a float32
reference rounded to BF16.

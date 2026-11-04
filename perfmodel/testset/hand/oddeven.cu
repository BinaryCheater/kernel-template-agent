// Odd-even transposition on shared memory inside a block.
__global__ void __launch_bounds__(128) k_oddeven(float* out, const float* in) {
  __shared__ float s[128];
  const int t = threadIdx.x, base = blockIdx.x * 128;
  s[t] = in[base + t];
  __syncthreads();
  for (int k = 0; k < 16; ++k) {
    const int j = (t & ~1) + (k & 1);
    if (j + 1 < 128 && (t == j || t == j + 1)) {
      const float a = s[j], b = s[j + 1];
      s[t] = (t == j) ? fminf(a, b) : fmaxf(a, b);
    }
    __syncthreads();
  }
  out[base + t] = s[t];
}

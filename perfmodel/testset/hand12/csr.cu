// Sparse matrix row sums: column indices, then values at them, then a store.
__global__ void __launch_bounds__(256) k_csr(float* y, const float* v, const unsigned* col) {
  const unsigned r = blockIdx.x * 256 + threadIdx.x;
  float s = 0.f;
  for (int k = 0; k < 8; ++k) {
    unsigned c = col[(r * 8 + k) & 4194303u] & 2097151u;
    s += v[c];
  }
  y[r] = s;
}

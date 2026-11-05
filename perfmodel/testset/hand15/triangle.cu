// Packed triangular matrix: row i, column j at i (i + 1) / 2 + j.
__global__ void __launch_bounds__(256) k_triangle(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  const unsigned i = t % 4093;
  unsigned s = 0;
  for (unsigned j = 0; j < 16; ++j) s += a[i * (i + 1) / 2 + (j * i) / 16];
  o[t] = s;
}

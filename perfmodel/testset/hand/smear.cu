// Each thread rewrites a window that overlaps its neighbour's, offset within the line.
__global__ void __launch_bounds__(256) k_smear(unsigned* out, const unsigned* in) {
  const size_t base = (size_t)blockIdx.x * 256 + threadIdx.x + 5;
  unsigned s = in[threadIdx.x];
  for (int k = 0; k < 24; ++k) {
    s = s * 2654435761u + k;
    out[base + (size_t)k * 4] = s;
  }
}

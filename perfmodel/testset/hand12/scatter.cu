// Random stores past the L2, no reads.
__global__ void __launch_bounds__(256) k_scatter(unsigned* a) {
  const size_t base = (size_t)blockIdx.x * 256 * 16 + threadIdx.x;
  for (int k = 0; k < 16; ++k) {
    unsigned h = (unsigned)(base + (size_t)k * 256) * 2654435761u;
    a[(h >> 7) & 8388607u] = h;
  }
}

// Histogram by global atomics.
__global__ void __launch_bounds__(256) k_hist(unsigned* h, const unsigned* in) {
  const size_t base = (size_t)blockIdx.x * 256 * 16 + threadIdx.x;
  for (int k = 0; k < 16; ++k) {
    unsigned v = in[base + (size_t)k * 256];
    atomicAdd(&h[(v ^ (v >> 11)) & 1023], 1u);
  }
}

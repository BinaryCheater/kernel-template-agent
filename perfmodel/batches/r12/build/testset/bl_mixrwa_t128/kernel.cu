// Read, count, write back: one load, one atomic, one store per thread pass.
extern "C" __global__ void __launch_bounds__(128) k_mixrwa(unsigned* h, unsigned* a, const unsigned* b) {
  const size_t base = (size_t)blockIdx.x * 128 * 16 + threadIdx.x;
  for (int k = 0; k < 16; ++k) {
    size_t i = base + (size_t)k * 128;
    unsigned v = b[i];
    atomicAdd(&h[(v ^ (v >> 7)) & 4095u], 1u);
    a[i] = v ^ k;
  }
}

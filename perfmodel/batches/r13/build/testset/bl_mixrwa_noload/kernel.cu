// Read, count, write back: one load, one atomic, one store per thread pass.
extern "C" __global__ void __launch_bounds__(256) k_mixrwa(unsigned* h, unsigned* a, const unsigned* b) {
  const size_t base = (size_t)blockIdx.x * 256 * 16 + threadIdx.x;
  for (int k = 0; k < 16; ++k) {
    size_t i = base + (size_t)k * 256;
    unsigned v = (unsigned)i * 2654435761u;
    atomicAdd(&h[(v ^ (v >> 7)) & 4095u], 1u);
    a[i] = v ^ k;
  }
}

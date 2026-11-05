// Stores under a data-dependent predicate: half the lanes write.
extern "C" __global__ void __launch_bounds__(256) k_maskstore(unsigned* a, const unsigned* b) {
  const size_t base = (size_t)blockIdx.x * 256 * 32 + threadIdx.x;
  for (int k = 0; k < 32; ++k) {
    size_t i = base + (size_t)k * 256;
    unsigned v = b[i];
    if (v & 256u) a[i] = v;
  }
}

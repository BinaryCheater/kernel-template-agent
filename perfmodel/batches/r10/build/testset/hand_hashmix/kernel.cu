// Integer mixing of a stream: rotates, multiplies, population counts, leading zeros.
__global__ void __launch_bounds__(256) k_hashmix(unsigned* out, const unsigned* in) {
  const size_t base = (size_t)blockIdx.x * 256 * 32 + threadIdx.x;
  unsigned h = 2166136261u;
  for (int k = 0; k < 32; ++k) {
    unsigned v = in[base + (size_t)k * 256];
    h = (h ^ v) * 16777619u;
    h = (h << 13) | (h >> 19);
    h += __popc(v) + __clz(v | 1u);
  }
  out[blockIdx.x * 256 + threadIdx.x] = h;
}

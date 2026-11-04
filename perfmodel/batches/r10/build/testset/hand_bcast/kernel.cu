// Warp shuffles and votes: a prefix sum over lanes, then a lane-wide vote.
__global__ void __launch_bounds__(256) k_bcast(unsigned* out, const unsigned* in) {
  const size_t base = (size_t)blockIdx.x * 256 * 16 + threadIdx.x;
  unsigned s = 0;
  for (int k = 0; k < 16; ++k) {
    unsigned v = in[base + (size_t)k * 256] & 255;
    for (int d = 1; d < 32; d *= 2) v += __shfl_up_sync(0xffffffffu, v, d);
    s += v + (unsigned)__any_sync(0xffffffffu, v > 100);
  }
  out[blockIdx.x * 256 + threadIdx.x] = s;
}

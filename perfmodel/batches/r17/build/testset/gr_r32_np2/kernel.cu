// round 17: GEMV's load: each warp a row, 4 bytes per lane, pitch 1344
extern "C" __global__ void __launch_bounds__(64) k_mem(unsigned* o, const char* a) {
  const unsigned row = blockIdx.x * 2 + threadIdx.y, kr = threadIdx.x;
  unsigned s = 0;
  for (unsigned ko = 0; ko < 11; ++ko) {
    const size_t at = (size_t)row * 1344 + (ko * 32 + kr) * 4;
    s ^= *(const unsigned*)(a + at);
  }
  if (kr == 0) o[row & 65535u] = s;
}

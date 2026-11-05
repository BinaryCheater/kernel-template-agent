// round 18: rows of 256 bytes, 16 bytes per lane, pitch 1296: decode's copies as loads
extern "C" __global__ void __launch_bounds__(128) k_mem(unsigned* o, const char* a) {
  unsigned s = 0;
  for (unsigned k = 0; k < 8; ++k) {
    const unsigned e = (blockIdx.x * 8 + k) * 128 + threadIdx.x;
    const size_t at = (size_t)(e / 16) * 1296 + (e % 16) * 16;
    s ^= (*(const uint4*)(a + at)).x;
  }
  o[blockIdx.x & 65535u] = s;
}

// round 17: first sectors alone, as many bursts as blocks ask
extern "C" __global__ void __launch_bounds__(128) k_mem(unsigned* o, const uint2* a) {
  const unsigned p = blockIdx.x, side = 0;
  unsigned s = 0;
  for (unsigned k = 0; k < 16; ++k) {
    const uint2 x = a[(p * 128 + threadIdx.x) * 128 + k * 8 + side];
    s ^= x.x ^ x.y;
  }
  o[blockIdx.x & 65535u] = s;
}

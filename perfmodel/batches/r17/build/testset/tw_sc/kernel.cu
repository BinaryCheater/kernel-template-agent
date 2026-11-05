// round 17: twin sectors by blocks on two SMs
extern "C" __global__ void __launch_bounds__(128) k_mem(unsigned* o, const uint2* a) {
  const unsigned p = blockIdx.x / 2u, side = (blockIdx.x & 1u) * 4;
  unsigned s = 0;
  for (unsigned k = 0; k < 16; ++k) {
    const uint2 x = a[(p * 128 + threadIdx.x) * 128 + k * 8 + side];
    s ^= x.x ^ x.y;
  }
  o[blockIdx.x & 65535u] = s;
}

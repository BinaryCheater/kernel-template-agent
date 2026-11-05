// round 17: twin sectors by neighbouring warps of a block
extern "C" __global__ void __launch_bounds__(128) k_mem(unsigned* o, const uint2* a) {
  const unsigned l = threadIdx.x & 31u, w = threadIdx.x >> 5;
  const uint2 x = a[(blockIdx.x * 64 + (w >> 1) * 32 + l) * 8 + (w & 1u) * 4];
  o[blockIdx.x & 65535u] = x.x ^ x.y;
}

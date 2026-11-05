// round 17: twin sectors at T 16, no gap
extern "C" __global__ void __launch_bounds__(16) k_mem(unsigned* o, const uint2* a) {
  const unsigned t = blockIdx.x * 16 + threadIdx.x;
  const uint2 x = a[t * 8];
  unsigned s = t;
  
  const uint2 y = a[t * 8 + 4];
  o[blockIdx.x & 65535u] = x.x ^ y.y ^ s;
}

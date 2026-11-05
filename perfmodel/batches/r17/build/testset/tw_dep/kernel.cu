// round 17: the twin after the first sector's data
extern "C" __global__ void __launch_bounds__(128) k_mem(unsigned* o, const uint2* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  const uint2 x = a[t * 8];
  unsigned s = t;
  if (x.x == 0x7fff1234u) s = 1;
  const uint2 y = a[t * 8 + 4];
  o[blockIdx.x & 65535u] = x.x ^ y.y ^ s;
}

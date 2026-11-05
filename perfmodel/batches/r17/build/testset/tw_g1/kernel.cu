// round 17: twin sectors, 1 passes apart
extern "C" __global__ void __launch_bounds__(128) k_mem(unsigned* o, const uint2* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  const uint2 x = a[t * 8];
  unsigned s = t;
  for (unsigned d = 0; d < 1; ++d) { s += d; asm volatile("" : "+r"(s)); }
  const uint2 y = a[t * 8 + 4];
  o[blockIdx.x & 65535u] = x.x ^ y.y ^ s;
}

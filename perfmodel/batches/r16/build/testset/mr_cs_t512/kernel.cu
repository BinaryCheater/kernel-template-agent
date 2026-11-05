// round 16: 8 MiB in order, evict-first
extern "C" __global__ void __launch_bounds__(512) k_mem(unsigned* o, const int4* a) {
  const unsigned t = blockIdx.x * 512 + threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < 4; ++k) {
    const int4 x = __ldcs(a + t * 4 + k);
    s += x.x ^ x.y ^ x.z ^ x.w;
  }
  o[blockIdx.x & 65535u] = s;
}

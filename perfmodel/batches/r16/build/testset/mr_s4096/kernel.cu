// round 16: 8 MiB, lanes 4096 bytes apart
extern "C" __global__ void __launch_bounds__(128) k_mem(unsigned* o, const uint2* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < 1; ++k) {
    const uint2 x = a[t * 512];
    s += x.x ^ x.y;
  }
  o[blockIdx.x & 65535u] = s;
}

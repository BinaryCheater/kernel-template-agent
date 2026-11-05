// round 16: 4 MiB, 4 sets in turn
extern "C" __global__ void __launch_bounds__(128) k_mem(unsigned* o, const uint2* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < 8; ++k) {
    const uint2 x = a[t * 8 + k];
    s += x.x ^ x.y;
  }
  o[blockIdx.x & 65535u] = s;
}

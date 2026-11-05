// round 16: 48 blocks, 256 KiB each
extern "C" __global__ void __launch_bounds__(128) k_mem(unsigned* o, const uint2* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < 256; ++k) {
    const uint2 x = a[t * 256 + k];
    s += x.x ^ x.y;
  }
  o[blockIdx.x & 65535u] = s;
}

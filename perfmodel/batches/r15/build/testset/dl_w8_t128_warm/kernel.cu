// round 15: 8 MiB read in order, 8 bytes per lane, 8 loads per thread
extern "C" __global__ void __launch_bounds__(128) k_dl(unsigned* o, const uint2* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < 8; ++k) {
    const uint2 x = a[t * 8 + k];
    s += x.x ^ x.y;
  }
  o[blockIdx.x & 65535u] = s;
}

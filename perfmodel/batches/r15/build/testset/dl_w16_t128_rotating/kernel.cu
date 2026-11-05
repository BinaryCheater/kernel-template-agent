// round 15: 8 MiB read in order, 16 bytes per lane, 4 loads per thread
extern "C" __global__ void __launch_bounds__(128) k_dl(unsigned* o, const uint4* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < 4; ++k) {
    const uint4 x = a[t * 4 + k];
    s += x.x ^ x.y ^ x.z ^ x.w;
  }
  o[blockIdx.x & 65535u] = s;
}

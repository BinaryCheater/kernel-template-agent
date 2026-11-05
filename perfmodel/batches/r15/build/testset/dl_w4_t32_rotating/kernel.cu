// round 15: 8 MiB read in order, 4 bytes per lane, 16 loads per thread
extern "C" __global__ void __launch_bounds__(32) k_dl(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 32 + threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < 16; ++k) {
    const unsigned x = a[t * 16 + k];
    s += x;
  }
  o[blockIdx.x & 65535u] = s;
}

// round 16: rows of 4096 bytes, pitch 8 KiB
extern "C" __global__ void __launch_bounds__(32) k_mem(unsigned* o, const uint2* a) {
  const unsigned t = blockIdx.x * 32 + threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < 16; ++k) {
    const uint2 x = a[blockIdx.x * 1024 + threadIdx.x * 16 + k];
    s += x.x ^ x.y;
  }
  o[blockIdx.x & 65535u] = s;
}

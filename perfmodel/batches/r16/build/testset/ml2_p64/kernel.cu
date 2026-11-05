// round 16: 8 passes of 64 KiB, two blocks per SM
extern "C" __global__ void __launch_bounds__(128) k_mem(unsigned* o, const uint4* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < 256; ++k) {
    const uint4 x = a[blockIdx.x * 4096 + threadIdx.x * 32 + k % 32u];
    s += x.x ^ x.y ^ x.z ^ x.w;
  }
  o[blockIdx.x & 65535u] = s;
}

// round 16: 8 passes of 192 KiB, a block per SM
extern "C" __global__ void __launch_bounds__(256) k_mem(unsigned* o, const uint4* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < 384; ++k) {
    const uint4 x = a[blockIdx.x * 12288 + threadIdx.x * 48 + k % 48u];
    s += x.x ^ x.y ^ x.z ^ x.w;
  }
  o[blockIdx.x & 65535u] = s;
}

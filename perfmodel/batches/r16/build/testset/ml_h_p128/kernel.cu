// round 16: 8 passes hashed over 128 KiB
extern "C" __global__ void __launch_bounds__(256) k_mem(unsigned* o, const uint4* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < 256; ++k) {
    const uint4 x = a[blockIdx.x * 8192 + ((threadIdx.x + k) * 2654435761u >> 8) % 8192u];
    s += x.x ^ x.y ^ x.z ^ x.w;
  }
  o[blockIdx.x & 65535u] = s;
}

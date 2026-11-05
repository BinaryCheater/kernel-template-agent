// round 16: hashed through 32 MiB
extern "C" __global__ void __launch_bounds__(512) k_mem(unsigned* o, const uint2* a) {
  const unsigned t = blockIdx.x * 512 + threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < 8; ++k) {
    const uint2 x = a[((t * 8 + k) * 2654435761u) >> 7 & 4194303u];
    s += x.x ^ x.y;
  }
  o[blockIdx.x & 65535u] = s;
}

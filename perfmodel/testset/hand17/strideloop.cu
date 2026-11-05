// Six loads per thread, 1284 bytes apart, over 16 MiB.
__global__ void __launch_bounds__(256) k_strideloop(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < 6; ++k) s ^= a[(t * 321 + k * 2053) & 4194303u];
  o[t & 65535u] = s;
}

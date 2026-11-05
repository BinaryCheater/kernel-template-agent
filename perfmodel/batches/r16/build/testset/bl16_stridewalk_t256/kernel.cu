// Eight loads per thread, 516 bytes apart, over 8 MiB.
extern "C" __global__ void __launch_bounds__(256) k_stridewalk(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < 8; ++k) s += a[(t * 129 + k * 1031) & 2097151u];
  o[t & 65535u] = s;
}

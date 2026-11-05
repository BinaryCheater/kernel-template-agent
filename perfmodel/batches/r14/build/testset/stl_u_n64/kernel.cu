// round 14: 64 words of stack, read 256 times at an index u
extern "C" __global__ void __launch_bounds__(256) k_stl(unsigned* o) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  unsigned r[64];
  for (unsigned i = 0; i < 64; ++i) r[i] = i * t;
  unsigned s = 0;
  for (unsigned k = 0; k < 256; ++k) s += r[(k * 5u) & 63u];
  o[t] = s;
}

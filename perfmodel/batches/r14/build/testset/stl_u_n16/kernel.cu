// round 14: 16 words of stack, read 64 times at an index u
extern "C" __global__ void __launch_bounds__(256) k_stl(unsigned* o) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  unsigned r[16];
  for (unsigned i = 0; i < 16; ++i) r[i] = i * t;
  unsigned s = 0;
  for (unsigned k = 0; k < 64; ++k) s += r[(k * 5u) & 15u];
  o[t] = s;
}

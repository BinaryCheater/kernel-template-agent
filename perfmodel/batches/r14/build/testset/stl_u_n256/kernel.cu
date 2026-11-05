// round 14: 256 words of stack, read 1024 times at an index u
extern "C" __global__ void __launch_bounds__(256) k_stl(unsigned* o) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  unsigned r[256];
  for (unsigned i = 0; i < 256; ++i) r[i] = i * t;
  unsigned s = 0;
  for (unsigned k = 0; k < 1024; ++k) s += r[(k * 5u) & 255u];
  o[t] = s;
}

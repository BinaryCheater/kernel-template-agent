// round 14: 1024 words of stack, read 4096 times at an index u
extern "C" __global__ void __launch_bounds__(256) k_stl(unsigned* o) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  unsigned r[1024];
  for (unsigned i = 0; i < 1024; ++i) r[i] = i * t;
  unsigned s = 0;
  for (unsigned k = 0; k < 4096; ++k) s += r[(k * 5u) & 1023u];
  o[t] = s;
}

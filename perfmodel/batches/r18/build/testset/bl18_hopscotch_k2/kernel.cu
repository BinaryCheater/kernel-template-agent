// Three hops through a table of 1 MiB.
extern "C" __global__ void __launch_bounds__(128) k_hopscotch(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  unsigned i = t;
  for (unsigned k = 0; k < 2; ++k) i = a[(i * 31u + k) & 262143u];
  o[t & 65535u] = i;
}

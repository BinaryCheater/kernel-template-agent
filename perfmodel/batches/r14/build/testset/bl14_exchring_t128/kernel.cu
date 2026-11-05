// Every lane swaps its id into a ring slot and keeps what it found.
extern "C" __global__ void __launch_bounds__(128) k_exchring(unsigned* o, unsigned* s) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  unsigned v = t;
  for (unsigned k = 0; k < 4; ++k) v = atomicExch(&s[(t * 33u + k) & 65535u], v);
  o[t] = v;
}

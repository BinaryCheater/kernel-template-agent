// A window of 48 words on the stack, an index per lane.
__global__ void __launch_bounds__(256) k_stackwin(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  unsigned w[48];
  for (unsigned k = 0; k < 48; ++k) w[k] = a[(t + k) & 262143u];
  unsigned s = 0;
  for (unsigned k = 0; k < 16; ++k) s += w[(t + k * 7) % 48u];
  o[t] = s;
}

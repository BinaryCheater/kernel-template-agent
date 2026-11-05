// Twelve loads kept on the stack, folded from both ends.
extern "C" __global__ void __launch_bounds__(128) k_stackfold(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  unsigned w[8];
  for (unsigned k = 0; k < 8; ++k) w[k] = a[(t * 13 + k * 97) & 262143u];
  unsigned s = 0;
  for (unsigned k = 0; k < 4; ++k) s += w[k] ^ w[7 - k];
  o[t] = s;
}

// Twelve loads kept on the stack, folded from both ends.
__global__ void __launch_bounds__(128) k_stackfold(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  unsigned w[12];
  for (unsigned k = 0; k < 12; ++k) w[k] = a[(t * 13 + k * 97) & 262143u];
  unsigned s = 0;
  for (unsigned k = 0; k < 6; ++k) s += w[k] ^ w[11 - k];
  o[t] = s;
}

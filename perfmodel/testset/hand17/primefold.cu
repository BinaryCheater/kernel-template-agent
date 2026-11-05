// Loads of a prime count of words, folded in a ring of stores.
__global__ void __launch_bounds__(128) k_primefold(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  for (unsigned k = 0; k < 3; ++k) o[(t * 11 + k * 7) % 524269u] = a[(t + k) % 131071u];
}

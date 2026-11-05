// Pairs of words copied with the two swapped.
__global__ void __launch_bounds__(128) k_swapwords(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  o[t * 2] = a[t * 2 + 1];
  o[t * 2 + 1] = a[t * 2];
}

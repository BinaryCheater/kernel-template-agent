// A word at a multiplied index.
__global__ void __launch_bounds__(128) k_lanehash(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  o[t] = a[(t * 2654435761u) >> 14];
}

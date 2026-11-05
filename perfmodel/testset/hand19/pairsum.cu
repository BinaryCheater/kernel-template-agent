// Two neighbouring words summed, each thread its own pair.
__global__ void __launch_bounds__(128) k_pairsum(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  o[t] = a[t * 2] + a[t * 2 + 1];
}

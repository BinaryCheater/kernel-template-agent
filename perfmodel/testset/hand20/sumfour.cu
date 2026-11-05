// Four words sixteen bytes apart.
__global__ void __launch_bounds__(128) k_sumfour(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  o[t] = a[t & 262143u] + a[(t + 4u) & 262143u] + a[(t + 8u) & 262143u] + a[(t + 12u) & 262143u];
}

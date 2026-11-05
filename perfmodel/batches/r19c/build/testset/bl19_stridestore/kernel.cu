// Stores five words apart, wrapping at 4 MiB.
extern "C" __global__ void __launch_bounds__(128) k_stridestore(unsigned* o) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  o[(t * 5) & 1048575u] = t;
}

// Stores at a linear congruence of the thread.
__global__ void __launch_bounds__(128) k_addput(unsigned* o) {
  unsigned x = blockIdx.x * 128 + threadIdx.x;
  for (unsigned k = 0; k < 3; ++k) {
    x = x * 1664525u + 1013904223u;
    o[x >> 12] = k;
  }
}

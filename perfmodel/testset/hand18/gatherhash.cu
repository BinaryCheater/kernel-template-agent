// A load at a hash of the thread, then a load through it.
__global__ void __launch_bounds__(128) k_gatherhash(unsigned* o, const unsigned* a, const unsigned* b) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  const unsigned i = a[(t * 2654435761u) >> 12];
  o[t & 65535u] = b[i & 262143u];
}

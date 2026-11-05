// round 14: blocks of 128 threads, 6 per SM on average: one warp chases loads, the rest wait
extern "C" __global__ void __launch_bounds__(128) k_occ(unsigned* o, const unsigned* a) {
  unsigned x = blockIdx.x * 128 + threadIdx.x;
  if (threadIdx.x < 32)
    for (int i = 0; i < 64; ++i) x = a[(x * 2654435761u >> 12) & 1048575u];
  __syncthreads();
  o[blockIdx.x & 65535u] = x;
}

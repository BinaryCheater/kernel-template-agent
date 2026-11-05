// round 16: 4 MiB read and written in place
extern "C" __global__ void __launch_bounds__(128) k_mem(unsigned* o) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  for (unsigned k = 0; k < 8; ++k) {
    o[t * 8 + k] += 1u;
  }
}

// round 16: 16 MiB, lanes 4 KiB apart
extern "C" __global__ void __launch_bounds__(128) k_mem(unsigned* o) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  for (unsigned k = 0; k < 1; ++k) {
    o[t * 1024] = t;
  }
}

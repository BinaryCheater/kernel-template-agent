// round 16: rows warm, loads then stores
extern "C" __global__ void __launch_bounds__(128) k_mem(unsigned* o, const uint2* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  for (unsigned k = 0; k < 4; ++k) {
    ((uint2*)o)[blockIdx.x * 512 + threadIdx.x * 4 + k] = a[blockIdx.x * 512 + threadIdx.x * 4 + k];
  }
}

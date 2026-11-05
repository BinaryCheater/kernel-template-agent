// round 16: rows warm, loads then stores
extern "C" __global__ void __launch_bounds__(256) k_mem(unsigned* o, const uint2* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (unsigned k = 0; k < 2; ++k) {
    ((uint2*)o)[blockIdx.x * 512 + threadIdx.x * 2 + k] = a[blockIdx.x * 512 + threadIdx.x * 2 + k];
  }
}

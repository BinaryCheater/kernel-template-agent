// Fill past the L2, no reads: 64 MiB of stores, contiguous lanes.
extern "C" __global__ void __launch_bounds__(128) k_memset(float* a) {
  const size_t base = (size_t)blockIdx.x * 128 * 64 + threadIdx.x;
  for (int k = 0; k < 64; ++k) a[base + (size_t)k * 128] = (float)k;
}

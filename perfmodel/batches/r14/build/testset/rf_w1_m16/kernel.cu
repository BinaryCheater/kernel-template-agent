// round 13: 1 of the 16 words of a 64-byte record per thread, over 16 MiB
extern "C" __global__ void __launch_bounds__(256) k_rf(unsigned* a) {
  const size_t r = (size_t)blockIdx.x * 256 + threadIdx.x;
  for (int k = 0; k < 1; ++k) a[r * 16 + k] = (unsigned)r + k;
}

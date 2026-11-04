// round 12b: 64 MiB of stores over 8 MiB
extern "C" __global__ void __launch_bounds__(256) k_wf(float* a) {
  const size_t n = 2097152ull;
  const size_t base = (size_t)blockIdx.x * 256 * 64 + threadIdx.x;
  for (int k = 0; k < 64; ++k) a[(base + (size_t)k * 256) % n] = (float)k;
}

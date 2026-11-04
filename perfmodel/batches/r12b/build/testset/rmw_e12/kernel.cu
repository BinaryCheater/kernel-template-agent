// round 12b: 12 MiB, y[i] += 2.5f * x[i];
extern "C" __global__ void __launch_bounds__(256) k_rmw(float* y, const float* x) {
  const size_t base = (size_t)blockIdx.x * 256 * 32 + threadIdx.x;
  for (int k = 0; k < 32; ++k) {
    size_t i = base + (size_t)k * 256;
    y[i] += 2.5f * x[i];
  }
}

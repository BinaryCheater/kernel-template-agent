// y += a x past the L2: the written lines were just read.
extern "C" __global__ void __launch_bounds__(128) k_axpy(float* y, const float* x) {
  const size_t base = (size_t)blockIdx.x * 128 * 32 + threadIdx.x;
  for (int k = 0; k < 32; ++k) {
    size_t i = base + (size_t)k * 128;
    y[i] += 2.5f * x[i];
  }
}

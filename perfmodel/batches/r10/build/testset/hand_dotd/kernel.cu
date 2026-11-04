// FP64 dot product per block, reduced by thread.
__global__ void __launch_bounds__(256) k_dotd(double* out, const double* a, const double* b) {
  const size_t base = (size_t)blockIdx.x * 256 * 32 + threadIdx.x;
  double s = 0.0;
  for (int k = 0; k < 32; ++k) {
    size_t i = base + (size_t)k * 256;
    s = fma(a[i], b[i], s);
  }
  out[blockIdx.x * 256 + threadIdx.x] = s;
}

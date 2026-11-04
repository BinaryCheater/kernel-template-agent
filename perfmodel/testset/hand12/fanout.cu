// One read, sixteen strided stores: a store-heavy loop inside the L2.
__global__ void __launch_bounds__(256) k_fanout(float* a, const float* b) {
  const size_t base = (size_t)blockIdx.x * 256 + threadIdx.x;
  float v = b[base];
  for (int k = 0; k < 16; ++k) a[base * 8 + (size_t)k * 1024] = v + k;
}

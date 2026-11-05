// Image tally on a 2-D grid: a load per thread, a bin atomic, a row of stores.
extern "C" __global__ void __launch_bounds__(256) k_tally2d(unsigned* h, float* a, const float* b) {
  const int w = 16 * gridDim.x;
  const int x = blockIdx.x * 16 + threadIdx.x, y = blockIdx.y * 16 + threadIdx.y;
  float v = b[y * w + x] + b[y * w + (x ^ 1)];
  atomicAdd(&h[(unsigned)(v * 16.f) & 255u], 1u);
  for (int k = 0; k < 4; ++k) a[(size_t)(y * w + x) * 4 + k] = v;
}

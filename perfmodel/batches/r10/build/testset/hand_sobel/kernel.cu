// Sobel on an 8-bit image: narrow loads, conversions.
__global__ void __launch_bounds__(256) k_sobel(float* out, const unsigned char* in) {
  const int n = 2048;
  const int x = blockIdx.x * 16 + threadIdx.x + 1, y = blockIdx.y * 16 + threadIdx.y + 1;
  const unsigned char* p = in + (size_t)y * n + x;
  const float gx = (float)p[-n + 1] + 2.f * p[1] + p[n + 1] - p[-n - 1] - 2.f * p[-1] - p[n - 1];
  const float gy = (float)p[n - 1] + 2.f * p[n] + p[n + 1] - p[-n - 1] - 2.f * p[-n] - p[-n + 1];
  out[y * n + x] = sqrtf(gx * gx + gy * gy);
}

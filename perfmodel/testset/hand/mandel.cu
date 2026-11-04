// Mandelbrot: lanes leave at different passes, no memory but the result.
__global__ void __launch_bounds__(256) k_mandel(unsigned* out) {
  const int x = blockIdx.x * 16 + threadIdx.x, y = blockIdx.y * 16 + threadIdx.y;
  const float cr = -2.f + x * (3.f / 512.f), ci = -1.5f + y * (3.f / 512.f);
  float zr = 0.f, zi = 0.f;
  unsigned n = 0;
  for (int k = 0; k < 64; ++k) {
    const float a = zr * zr, b = zi * zi;
    if (a + b > 4.f) break;
    zi = 2.f * zr * zi + ci;
    zr = a - b + cr;
    n++;
  }
  out[y * 512 + x] = n;
}

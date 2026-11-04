// Five-point stencil on a 2048 x 2048 fp32 grid, 16 x 16 thread blocks.
extern "C" __global__ void __launch_bounds__(256) k_stencil(float* out, const float* in) {
  const int n = 2048;
  const int x = blockIdx.x * 16 + threadIdx.x, y = blockIdx.y * 16 + threadIdx.y;
  const int xm = x > 0 ? x - 1 : x, xp = x < n - 1 ? x + 1 : x;
  const int ym = y > 0 ? y - 1 : y, yp = y < n - 1 ? y + 1 : y;
  out[y * n + x] = 0.2f * (in[y * n + x] + in[y * n + xm] + in[y * n + xp] + in[ym * n + x] + in[yp * n + x]);
}

// A five-point stencil on a periodic grid of 1536 x 1365: remainders at both axes.
extern "C" __global__ void __launch_bounds__(256) k_wrapstencil(float* o, const float* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x, W = 1536, H = 1365;
  const unsigned y = t / W, x = t - y * W;
  const unsigned up = (y + H - 1) % H, dn = (y + 1) % H, lf = (x + W - 1) % W, rt = (x + 1) % W;
  o[y * W + x] = a[up * W + x] + a[dn * W + x] + a[y * W + lf] + a[y * W + rt] - 4 * a[y * W + x];
}

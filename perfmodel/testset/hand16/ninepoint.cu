// A nine-point stencil, no remainders.
__global__ void __launch_bounds__(256) k_ninepoint(float* o, const float* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x, W = 1024;
  const unsigned y = t / W + 1, x = t % (W - 2) + 1;
  float s = 0;
  for (int d = -1; d <= 1; ++d) s += a[(y + d) * W + x - 1] + a[(y + d) * W + x] + a[(y + d) * W + x + 1];
  o[y * W + x] = s;
}

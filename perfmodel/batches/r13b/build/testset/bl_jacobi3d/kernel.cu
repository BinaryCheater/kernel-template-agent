// Seven-point stencil on a 128 cube, one z plane per block.
extern "C" __global__ void __launch_bounds__(256) k_jacobi3d(float* a, const float* b) {
  const int N = 128;
  const int t = blockIdx.x * 256 + threadIdx.x;
  const int x = t & (N - 1), y = (t >> 7) & (N - 1), z = (t >> 14) & (N - 1);
  const size_t i = ((size_t)z * N + y) * N + x;
  float s = b[i];
  if (x > 0 && x < N - 1) s += b[i - 1] + b[i + 1];
  if (y > 0 && y < N - 1) s += b[i - N] + b[i + N];
  if (z > 0 && z < N - 1) s += b[i - N * N] + b[i + N * N];
  a[i] = s / 7.f;
}

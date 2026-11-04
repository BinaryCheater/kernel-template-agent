// Clear a 2048 x 2048 image tile by tile: rows of a tile are a whole image row apart.
extern "C" __global__ void __launch_bounds__(256) k_tilefill(float* a) {
  const unsigned x0 = blockIdx.x * 64 + threadIdx.x * 4;
  const unsigned y0 = blockIdx.y * 64 + threadIdx.y * 4;
  for (unsigned j = 0; j < 4; ++j)
    for (unsigned i = 0; i < 4; ++i) a[(y0 + j) * 2048u + x0 + i] = 0.5f;
}

// Stores at an index divided by a run-time count: the block count of the launch.
__global__ void __launch_bounds__(256) k_divrun(unsigned* o) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  const unsigned d = gridDim.x / 3 + 1;
  for (unsigned k = 0; k < 8; ++k) o[((t + k * 256) / d * 64u + t % d * 16u + k) & 4194303u] = t;
}

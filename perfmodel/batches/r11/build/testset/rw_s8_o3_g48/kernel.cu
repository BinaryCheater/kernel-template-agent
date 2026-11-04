// round 11: stores 8 elements apart per thread pass, first lane 3 off a line
extern "C" __global__ void __launch_bounds__(256) k_rw(unsigned* out) {
  const size_t base = (size_t)blockIdx.x * 4096 + threadIdx.x + 3;
  for (int k = 0; k < 16; ++k)
    out[base + (size_t)k * 8] = k;
}

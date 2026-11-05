// Each lane counts its loaded word in one of 256 bins.
extern "C" __global__ void __launch_bounds__(256) k_histlane(unsigned* h, const unsigned* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (unsigned k = 0; k < 8; ++k) atomicAdd(&h[a[(t * 8 + k) & 1048575u] & 255u], 1u);
}

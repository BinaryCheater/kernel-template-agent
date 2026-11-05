// A tally of 256 bins by the low byte of a word loaded.
__global__ void __launch_bounds__(128) k_tallybyte(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  atomicAdd(&o[a[t & 262143u] & 255u], 1u);
}

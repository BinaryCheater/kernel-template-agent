// A histogram of 1024 bins by a byte pair of each loaded word.
__global__ void __launch_bounds__(256) k_histbytes(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  atomicAdd(&o[(a[t & 262143u] >> 6) & 1023u], 1u);
}

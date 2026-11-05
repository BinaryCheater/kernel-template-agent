// A block of two rows of 64 threads, each row reads one run of words and stores its sum.
__global__ void __launch_bounds__(128) k_rowthree(unsigned* o, const unsigned* a) {
  const unsigned r = blockIdx.x * 2 + threadIdx.y;
  o[r * 64 + threadIdx.x] = a[r * 64 + threadIdx.x] + a[(r + 1) * 64 + threadIdx.x];
}

// A block of four rows of 32 threads, each row its own run of words.
__global__ void __launch_bounds__(128) k_twodim(unsigned* o, const unsigned* a) {
  const unsigned r = blockIdx.x * 4 + threadIdx.y;
  o[r * 32 + threadIdx.x] = a[r * 32 + threadIdx.x] + r;
}

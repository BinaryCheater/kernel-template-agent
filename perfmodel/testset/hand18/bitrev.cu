// A copy to the bit-reversed index.
__global__ void __launch_bounds__(128) k_bitrev(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  o[__brev(t) >> 12] = a[t];
}

// Records of three words read whole, one word out.
__global__ void __launch_bounds__(128) k_triplet(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  o[t] = a[t * 3] * a[t * 3 + 1] * a[t * 3 + 2];
}

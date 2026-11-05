// Three of six fields of a record of 24 bytes.
__global__ void __launch_bounds__(256) k_recordthree(unsigned* o) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  o[t * 6] = t;
  o[t * 6 + 2] = t + 1;
  o[t * 6 + 5] = t + 2;
}

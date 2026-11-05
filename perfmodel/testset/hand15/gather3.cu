// Three fields of a record of 28 bytes.
__global__ void __launch_bounds__(256) k_gather3(float* o, const float* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  const float* r = a + t * 7;
  o[t] = r[0] * r[3] + r[6];
}

// Five of nine fields of a record of 36 bytes.
__global__ void __launch_bounds__(128) k_recordfive(unsigned* o) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  o[t * 9] = t;
  o[t * 9 + 1] = t + 1;
  o[t * 9 + 3] = t + 2;
  o[t * 9 + 6] = t + 3;
  o[t * 9 + 8] = t + 4;
}

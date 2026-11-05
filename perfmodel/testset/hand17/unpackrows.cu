// Packed rows of 128 words read, written at a pitch of 512 words.
__global__ void __launch_bounds__(128) k_unpackrows(unsigned* o, const unsigned* a) {
  const unsigned t = threadIdx.x, y = blockIdx.x;
  o[y * 512 + t] = a[y * 128 + t] ^ y;
}

// Rows of 256 words read at a pitch of 1024 words, written packed.
extern "C" __global__ void __launch_bounds__(256) k_packrows(unsigned* o, const unsigned* a) {
  const unsigned t = threadIdx.x, y = blockIdx.x;
  o[y * 256 + t] = a[y * 1024 + t] + 1;
}

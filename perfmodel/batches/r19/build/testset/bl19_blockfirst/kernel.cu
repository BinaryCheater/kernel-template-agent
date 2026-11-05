// The first thread of a block alone loads four words and stores one.
extern "C" __global__ void __launch_bounds__(128) k_blockfirst(unsigned* o, const unsigned* a) {
  if (threadIdx.x != 0) return;
  unsigned s = 0;
  for (unsigned k = 0; k < 4; ++k) s += a[(blockIdx.x * 64 + k * 16) & 262143u];
  o[blockIdx.x] = s;
}

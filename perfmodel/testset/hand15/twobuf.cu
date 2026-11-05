// A select between two buffers by the thread's parity.
__global__ void __launch_bounds__(256) k_twobuf(unsigned* o, const unsigned* a, const unsigned* b) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < 8; ++k) {
    const unsigned* p = (t & 1) ? a : b;
    s += p[(t * 16 + k) & 1048575u];
  }
  o[t] = s;
}

// Eight bytes summed, three bytes between threads.
extern "C" __global__ void __launch_bounds__(128) k_byteload(unsigned* o, const unsigned char* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < 8; ++k) s += a[(t * 3 + k) & 4194303u];
  o[t] = s;
}

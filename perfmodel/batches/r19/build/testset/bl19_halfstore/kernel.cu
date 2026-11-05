// Words narrowed to 16 bits on the way out.
extern "C" __global__ void __launch_bounds__(128) k_halfstore(unsigned short* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  o[t] = (unsigned short)(a[t] >> 3);
}

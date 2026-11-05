// Words read from the top of the buffer down.
extern "C" __global__ void __launch_bounds__(128) k_reverseread(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 128 + threadIdx.x;
  o[t] = a[262143u - (t & 262143u)] ^ t;
}

// Records of three words written whole, a word read beside.
extern "C" __global__ void __launch_bounds__(256) k_threestore(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  const unsigned v = a[t];
  o[t * 3] = v;
  o[t * 3 + 1] = v + 1;
  o[t * 3 + 2] = v + 2;
}

// Every thread reads one field of its 64-byte record and writes two others.
extern "C" __global__ void __launch_bounds__(256) k_recordpatch(unsigned* a) {
  const size_t r = (size_t)blockIdx.x * 256 + threadIdx.x;
  unsigned v = a[r * 16];
  a[r * 16 + 1] = v + (unsigned)r;
  a[r * 16 + 2] = v ^ (unsigned)r;
}

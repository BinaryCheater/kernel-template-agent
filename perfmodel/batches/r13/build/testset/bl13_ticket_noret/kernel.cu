// A shared append cursor: every thread takes a slot from one counter and fills it.
extern "C" __global__ void __launch_bounds__(256) k_ticket(unsigned* o, unsigned* c) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (unsigned k = 0; k < 4; ++k) {
    unsigned n = atomicAdd(&c[0], 1u);
  }
}

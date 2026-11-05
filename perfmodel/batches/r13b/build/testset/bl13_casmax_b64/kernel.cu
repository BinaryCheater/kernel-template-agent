// One attempt per item to raise a hashed maximum by compare-and-swap.
extern "C" __global__ void __launch_bounds__(256) k_casmax(unsigned* w) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (unsigned k = 0; k < 8; ++k) {
    unsigned x = (t + k * 4099u) * 2654435761u;
    unsigned j = (x >> 9) & 63u;
    unsigned old = w[j];
    if (x > old) atomicCAS(&w[j], old, x);
  }
}

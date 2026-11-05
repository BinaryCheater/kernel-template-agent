// Lanes xor into hashed words of a large table.
__global__ void __launch_bounds__(256) k_xorwide(unsigned* w) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (unsigned k = 0; k < 8; ++k) {
    unsigned x = (t + k * 4099u) * 2654435761u;
    atomicXor(&w[(x >> 7) & 4194303u], t);
  }
}

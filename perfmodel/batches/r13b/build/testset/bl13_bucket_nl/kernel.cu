// Each block scatters its values into its own 16 KiB bucket.
extern "C" __global__ void __launch_bounds__(256) k_bucket(unsigned* o, const unsigned* a) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (unsigned k = 0; k < 8; ++k) {
    unsigned v = (t + k) * 2654435761u;
    o[blockIdx.x * 4096u + ((v * 2654435761u) >> 12 & 4095u)] = v;
  }
}

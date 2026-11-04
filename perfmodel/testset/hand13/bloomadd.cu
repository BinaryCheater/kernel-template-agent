// Set three hashed bytes per item in a 16 MiB byte map.
__global__ void __launch_bounds__(256) k_bloomadd(unsigned char* b) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (unsigned k = 0; k < 4; ++k) {
    unsigned x = (t + k * 4099u) * 2654435761u;
    b[(x >> 5) & 16777215u] = 1;
    b[((x ^ 0x5bd1e995u) * 2246822519u >> 5) & 16777215u] = 1;
    b[((x ^ 0x9e3779b9u) * 3266489917u >> 5) & 16777215u] = 1;
  }
}

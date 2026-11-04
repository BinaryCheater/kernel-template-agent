// Graph push: three reads of neighbours, atomic adds into their ranks past the L1.
__global__ void __launch_bounds__(256) k_rankpush(unsigned* r, const unsigned* adj) {
  const unsigned t = blockIdx.x * 256 + threadIdx.x;
  for (int k = 0; k < 8; ++k) {
    unsigned e = (t * 3u + k) & 1048575u;
    unsigned n0 = adj[e], n1 = adj[(e + 1) & 1048575u], n2 = adj[e ^ 4096u];
    atomicAdd(&r[(n0 ^ n1) & 262143u], n2 & 7u);
  }
}

// A Morton curve over 2048 x 2048 words: the address is built bit by bit.
extern "C" __global__ void __launch_bounds__(256) k_morton(unsigned* o, const unsigned* a) {
  unsigned x = blockIdx.x * 256 + threadIdx.x;
  unsigned s = 0;
  for (unsigned k = 0; k < 4; ++k) {
    unsigned p = (x + k * 16) & 2047u, q = (x >> 3) & 2047u;
    p = (p | p << 8) & 0x00ff00ffu; p = (p | p << 4) & 0x0f0f0f0fu;
    p = (p | p << 2) & 0x33333333u; p = (p | p << 1) & 0x55555555u;
    q = (q | q << 8) & 0x00ff00ffu; q = (q | q << 4) & 0x0f0f0f0fu;
    q = (q | q << 2) & 0x33333333u; q = (q | q << 1) & 0x55555555u;
    s += a[p | q << 1];
  }
  o[x] = s;
}

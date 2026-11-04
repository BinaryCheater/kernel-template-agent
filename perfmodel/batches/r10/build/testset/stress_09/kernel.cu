// generated, seed 1010, kernel 9
#define MASK ((1u << 22) - 1)
#define OMASK ((1u << 18) - 1)
__global__ void __launch_bounds__(256) k_stress9(unsigned* out, const unsigned* in) {
  __shared__ float s[64];
  const int t = threadIdx.x;
  const size_t base = (size_t)blockIdx.x * 256 + t;
  float f0 = t, f1 = 1.f, f2 = 2.f, f3 = 3.f;
  unsigned u0 = t, u1 = 1, u2 = 2, u3 = 3;
  for (int k = 0; k < 16; ++k) {
    u2 = (u0 ^ (u3 >> 7)) + (u2 << 3);
    out[(base + (size_t)k * 8) & OMASK] = u2;
    u1 = u1 * 2654435761u + u0;
    f2 = (float)u3 * 1e-9f;
    u1 = in[(base ^ u0) & MASK];
    s[t & 63] = f0;
  }
  s[t & 63] = f0 + f1 + f2 + f3;
  out[base & OMASK] = u0 ^ u1 ^ u2 ^ u3 ^ (unsigned)s[0];
}

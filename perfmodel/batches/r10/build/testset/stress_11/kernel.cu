// generated, seed 1010, kernel 11
#define MASK ((1u << 22) - 1)
#define OMASK ((1u << 18) - 1)
__global__ void __launch_bounds__(256) k_stress11(unsigned* out, const unsigned* in) {
  __shared__ float s[64];
  const int t = threadIdx.x;
  const size_t base = (size_t)blockIdx.x * 256 + t;
  float f0 = t, f1 = 1.f, f2 = 2.f, f3 = 3.f;
  unsigned u0 = t, u1 = 1, u2 = 2, u3 = 3;
  for (int k = 0; k < 16; ++k) {
    f1 = (u1 & 1) ? f1 : f3;
    u0 ^= in[(base + (size_t)k * 256) & MASK];
    u2 = in[(base ^ u1) & MASK];
    f1 = sqrtf(fabsf(f2) + 1.f);
    u3 = (unsigned)fabsf(f1) + u2;
    f1 += s[(t + k) & 63];
    f2 = expf(f3 * 0.001f);
    s[t & 63] = f3;
    u1 = (u3 ^ (u1 >> 7)) + (u1 << 3);
    f0 = f0 + f1;
    f0 += s[(t + k) & 63];
    f1 = f3 + f1;
    f3 += s[(t + k) & 63];
    u1 = (u2 ^ (u3 >> 7)) + (u1 << 3);
    out[(base + (size_t)k * 8) & OMASK] = u1;
  }
  s[t & 63] = f0 + f1 + f2 + f3;
  out[base & OMASK] = u0 ^ u1 ^ u2 ^ u3 ^ (unsigned)s[0];
}

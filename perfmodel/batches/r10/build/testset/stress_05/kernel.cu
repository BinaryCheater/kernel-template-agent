// generated, seed 1010, kernel 5
#define MASK ((1u << 22) - 1)
#define OMASK ((1u << 18) - 1)
extern "C" __global__ void __launch_bounds__(128) k_stress5(unsigned* out, const unsigned* in) {
  __shared__ float s[64];
  const int t = threadIdx.x;
  const size_t base = (size_t)blockIdx.x * 128 + t;
  float f0 = t, f1 = 1.f, f2 = 2.f, f3 = 3.f;
  unsigned u0 = t, u1 = 1, u2 = 2, u3 = 3;
  for (int k = 0; k < 8; ++k) {
    f1 = f0 * 0.999f + f1;
    f2 = f3 / (f3 + 1.5f);
    u1 = (u2 ^ (u1 >> 7)) + (u1 << 3);
    u2 ^= in[(base + (size_t)k * 64) & MASK];
    u3 = (u3 ^ (u1 >> 7)) + (u3 << 3);
    u2 = (unsigned)fabsf(f0) + u2;
    out[(base + (size_t)k * 8) & OMASK] = u0;
    f0 = (float)u3 * 1e-9f;
    f3 += s[(t + k) & 63];
    f1 = f1 / (f1 + 1.5f);
    u1 = (u3 ^ (u0 >> 7)) + (u1 << 3);
    u3 = in[(base ^ u3) & MASK];
  }
  s[t & 63] = f0 + f1 + f2 + f3;
  out[base & OMASK] = u0 ^ u1 ^ u2 ^ u3 ^ (unsigned)s[0];
}

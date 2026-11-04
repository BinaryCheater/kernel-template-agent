// generated, seed 1010, kernel 10
#define MASK ((1u << 22) - 1)
#define OMASK ((1u << 18) - 1)
extern "C" __global__ void __launch_bounds__(128) k_stress10(unsigned* out, const unsigned* in) {
  __shared__ float s[64];
  const int t = threadIdx.x;
  const size_t base = (size_t)blockIdx.x * 128 + t;
  float f0 = t, f1 = 1.f, f2 = 2.f, f3 = 3.f;
  unsigned u0 = t, u1 = 1, u2 = 2, u3 = 3;
  for (int k = 0; k < 8; ++k) {
    f2 = f0 + f1;
    f1 = expf(f2 * 0.001f);
    f3 = f1 / (f2 + 1.5f);
    f2 = (u2 & 1) ? f2 : f2;
    u2 = (unsigned)fabsf(f2) + u2;
    u3 = u2 * 2654435761u + u0;
    u3 = (u1 ^ (u2 >> 7)) + (u3 << 3);
    out[(base + (size_t)k * 64) & OMASK] = u0;
    f3 = sqrtf(fabsf(f1) + 1.f);
    u3 = (unsigned)fabsf(f0) + u1;
    u3 ^= in[(base + (size_t)k * 64) & MASK];
    f0 = sqrtf(fabsf(f0) + 1.f);
    f0 = expf(f1 * 0.001f);
    f3 += s[(t + k) & 63];
    f2 = sqrtf(fabsf(f2) + 1.f);
    f3 = (u2 & 1) ? f2 : f3;
  }
  s[t & 63] = f0 + f1 + f2 + f3;
  out[base & OMASK] = u0 ^ u1 ^ u2 ^ u3 ^ (unsigned)s[0];
}

// generated, seed 1010, kernel 4
#define MASK ((1u << 22) - 1)
#define OMASK ((1u << 18) - 1)
__global__ void __launch_bounds__(64) k_stress4(unsigned* out, const unsigned* in) {
  __shared__ float s[64];
  const int t = threadIdx.x;
  const size_t base = (size_t)blockIdx.x * 64 + t;
  float f0 = t, f1 = 1.f, f2 = 2.f, f3 = 3.f;
  unsigned u0 = t, u1 = 1, u2 = 2, u3 = 3;
  for (int k = 0; k < 32; ++k) {
    u3 ^= in[(base + (size_t)k * 256) & MASK];
    u2 = in[(base ^ u1) & MASK];
    u3 = u0 * 2654435761u + u3;
    out[(base + (size_t)k * 1) & OMASK] = u2;
    u0 = (unsigned)fabsf(f2) + u2;
    s[t & 63] = f2;
    f0 = expf(f1 * 0.001f);
    out[(base + (size_t)k * 1024) & OMASK] = u3;
    out[(base + (size_t)k * 256) & OMASK] = u3;
  }
  s[t & 63] = f0 + f1 + f2 + f3;
  out[base & OMASK] = u0 ^ u1 ^ u2 ^ u3 ^ (unsigned)s[0];
}

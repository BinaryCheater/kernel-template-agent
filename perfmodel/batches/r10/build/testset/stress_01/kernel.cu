// generated, seed 1010, kernel 1
#define MASK ((1u << 22) - 1)
#define OMASK ((1u << 18) - 1)
__global__ void __launch_bounds__(256) k_stress1(unsigned* out, const unsigned* in) {
  __shared__ float s[64];
  const int t = threadIdx.x;
  const size_t base = (size_t)blockIdx.x * 256 + t;
  float f0 = t, f1 = 1.f, f2 = 2.f, f3 = 3.f;
  unsigned u0 = t, u1 = 1, u2 = 2, u3 = 3;
  for (int k = 0; k < 16; ++k) {
    f1 = expf(f1 * 0.001f);
    f1 += (float)in[(base * 97 + (size_t)k * 4099) & MASK];
    u3 = in[(base ^ u2) & MASK];
    out[(base + (size_t)k * 1024) & OMASK] = u3;
    f3 = sqrtf(fabsf(f3) + 1.f);
    u3 = (u3 ^ (u0 >> 7)) + (u3 << 3);
    f0 = f2 * 0.999f + f0;
    u3 = u0 * 2654435761u + u0;
    u3 = (unsigned)fabsf(f2) + u0;
    u3 = u2 * 2654435761u + u3;
    f1 += (float)in[(base * 3 + (size_t)k * 4099) & MASK];
    f1 += (float)in[(base * 97 + (size_t)k * 4099) & MASK];
    f3 = (float)u2 * 1e-9f;
    f0 = f1 / (f0 + 1.5f);
    f0 = expf(f3 * 0.001f);
    u1 = u2 * 2654435761u + u3;
  }
  s[t & 63] = f0 + f1 + f2 + f3;
  out[base & OMASK] = u0 ^ u1 ^ u2 ^ u3 ^ (unsigned)s[0];
}

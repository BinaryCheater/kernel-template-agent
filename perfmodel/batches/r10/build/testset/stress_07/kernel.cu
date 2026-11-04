// generated, seed 1010, kernel 7
#define MASK ((1u << 22) - 1)
#define OMASK ((1u << 18) - 1)
extern "C" __global__ void __launch_bounds__(64) k_stress7(unsigned* out, const unsigned* in) {
  __shared__ float s[64];
  const int t = threadIdx.x;
  const size_t base = (size_t)blockIdx.x * 64 + t;
  float f0 = t, f1 = 1.f, f2 = 2.f, f3 = 3.f;
  unsigned u0 = t, u1 = 1, u2 = 2, u3 = 3;
  for (int k = 0; k < 32; ++k) {
    u3 = u0 * 2654435761u + u0;
    u2 ^= in[(base + (size_t)k * 8) & MASK];
    f3 += (float)in[(base * 97 + (size_t)k * 4099) & MASK];
    f1 = f1 + f1;
    f1 = f1 + f3;
    u1 = in[(base ^ u0) & MASK];
  }
  s[t & 63] = f0 + f1 + f2 + f3;
  out[base & OMASK] = u0 ^ u1 ^ u2 ^ u3 ^ (unsigned)s[0];
}

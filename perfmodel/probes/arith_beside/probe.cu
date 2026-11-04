// Arithmetic beside a chip-bound stream (question 27; RMSNorm compute removed: graph rows -20 %
// measured, replay +6 %). 128 threads: L loads of 32 B, then A FFMA, then S stores of 32 B per
// thread (a row kernel's shape), grids 96 and 512, warm. Does the arithmetic add to the span, or does
// it hide while the chip's L2 and store limits bind? Reported: span on an SM (cycles). The l2mix
// pattern.
#include "../common.cuh"
constexpr int BLOCK = 4096 * 8;
typedef unsigned long long u64;
template <int L, int S, int A>
__global__ void __launch_bounds__(128) k(u64* out, const u64* X, u64* Y, unsigned* sink) {
  u64 c0 = clock64();
  u64 s = threadIdx.x;
  char* base = (char*)Y + (size_t)blockIdx.x * BLOCK;
  const char* src = (const char*)X + (size_t)blockIdx.x * BLOCK;
#pragma unroll
  for (int i = 0; i < L; i++) {
    const u64* p = (const u64*)(src + i * 4096 + threadIdx.x * 32);
    u64 a, b, c, d;
    asm volatile("ld.global.nc.L2::256B.v4.u64 {%0,%1,%2,%3}, [%4];" : "=l"(a), "=l"(b), "=l"(c), "=l"(d) : "l"(p));
    s ^= a ^ b ^ c ^ d;
  }
  float f = (float)(unsigned)s;
#pragma unroll
  for (int i = 0; i < A; i++) f = fmaf(f, 0.9999f, 0.25f);
  s ^= (u64)(unsigned)(int)f;
#pragma unroll
  for (int i = 0; i < S; i++) {
    u64* p = (u64*)(base + i * 4096 + threadIdx.x * 32);
    asm volatile("st.global.v4.u64 [%0], {%1,%2,%3,%4};" : : "l"(p), "l"(s), "l"(s), "l"(s), "l"(s) : "memory");
  }
  __syncthreads();
  u64 c1 = clock64();
  if (threadIdx.x == 0) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    u64* o = out + 3 * blockIdx.x;
    o[0] = smid, o[1] = c0, o[2] = c1;
  }
  if (s == 12345) sink[0] = 1;
}
static double reduce(const std::vector<u64>& h, int nb) {
  std::vector<u64> lo(256, ~0ull), hi(256, 0);
  for (int b = 0; b < nb; b++) {
    int m = h[3 * b] % 256;
    lo[m] = std::min(lo[m], h[3 * b + 1]), hi[m] = std::max(hi[m], h[3 * b + 2]);
  }
  std::vector<double> span;
  for (int m = 0; m < 256; m++)
    if (hi[m]) span.push_back((double)(hi[m] - lo[m]));
  return stat_of(span).median;
}
int main() {
  u64 *o, *X, *Y;
  unsigned* sink;
  const size_t B = 512ull * BLOCK * 2 + 4096;
  CK(cudaMalloc(&o, 8 * 3 * 512));
  CK(cudaMalloc(&X, B));
  CK(cudaMalloc(&Y, B));
  CK(cudaMalloc(&sink, 4));
  CK(cudaMemset(X, 0, B));
  auto point = [&](auto kern, const char* name) {
    for (int nb : {96, 512}) {
      std::vector<u64> h(3 * nb);
      Stat s = repeat([&] {
        CK(event_us([&] { kern<<<nb, 128>>>(o, X, Y, sink); }) < 0);
        CK(cudaMemcpy(h.data(), o, 8 * 3 * nb, cudaMemcpyDeviceToHost));
        return reduce(h, nb);
      });
      char p[48];
      snprintf(p, sizeof p, "%s_g%d", name, nb);
      emit("arith_beside", p, s, "cycles");
    }
  };
#define PT(L, S, A) point(k<L, S, A>, "L" #L "_S" #S "_A" #A)
  PT(2, 0, 0); PT(2, 0, 32); PT(2, 0, 128); PT(2, 0, 512);
  PT(2, 2, 0); PT(2, 2, 32); PT(2, 2, 128); PT(2, 2, 512);
  PT(0, 2, 32); PT(0, 2, 128); PT(0, 2, 512);
}

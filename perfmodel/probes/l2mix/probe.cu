// What the chip L2 share depends on (open question 15; inventory §4 chip L2 share under stores): L
// loads and S stores per thread, 128 threads, warm, 32 bytes per load. A store of W bytes per lane,
// lanes D bytes apart: contiguous 32, 16, 4 (D = W), strided 16 and 32 at D = 64 (row_reduce's
// shape). P 0 interleaves loads and stores per thread, P 1 does all loads, then all stores (a row
// kernel's shape). Grids 96 and 512: the blocks of a row kernel. Reported: the span on an SM
// (cycles); share = (both - stores alone) / loads alone is derived from it. The copy of the st256
// pattern (reduce, kernel arguments).
#include "../common.cuh"
constexpr int BLOCK = 4096 * 8;  // bytes between blocks
typedef unsigned long long u64;
template <int W>
__device__ __forceinline__ void store(u64* p, u64 s) {
  if constexpr (W == 32)
    asm volatile("st.global.v4.u64 [%0], {%1,%2,%3,%4};" : : "l"(p), "l"(s), "l"(s), "l"(s), "l"(s) : "memory");
  else if constexpr (W == 16)
    asm volatile("st.global.v2.u64 [%0], {%1,%2};" : : "l"(p), "l"(s), "l"(s) : "memory");
  else
    asm volatile("st.global.u32 [%0], %1;" : : "l"(p), "r"((unsigned)s) : "memory");
}
template <int L, int S, int W, int D, int P>
__global__ void k(u64* out, const u64* X, u64* Y, unsigned* sink) {
  u64 c0 = clock64();
  u64 s = threadIdx.x;
  char* base = (char*)Y + (size_t)blockIdx.x * BLOCK;
  const char* src = (const char*)X + (size_t)blockIdx.x * BLOCK;
  auto ld = [&](int i) {
    const u64* p = (const u64*)(src + i * 4096 + threadIdx.x * 32);
    u64 a, b, c, d;
    asm volatile("ld.global.nc.L2::256B.v4.u64 {%0,%1,%2,%3}, [%4];" : "=l"(a), "=l"(b), "=l"(c), "=l"(d) : "l"(p));
    s ^= a ^ b ^ c ^ d;
  };
  auto st = [&](int i) { store<W>((u64*)(base + i * 128 * D + threadIdx.x * D), s); };
  if (P) {
#pragma unroll
    for (int i = 0; i < L; i++) ld(i);
#pragma unroll
    for (int i = 0; i < S; i++) st(i);
  } else {
    constexpr int N = L > S ? L : S;
#pragma unroll
    for (int i = 0; i < N; i++) {  // evenly spaced
      if ((i + 1) * L / N > i * L / N) ld(i * L / N);
      if ((i + 1) * S / N > i * S / N) st(i * S / N);
    }
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
      emit("l2mix", p, s, "cycles");
    }
  };
#define PT(L, S, W, D, P) point(k<L, S, W, D, P>, "L" #L "_S" #S "_W" #W "_D" #D "_P" #P)
  // alone
  PT(1, 0, 32, 32, 0); PT(2, 0, 32, 32, 0); PT(4, 0, 32, 32, 0);
  PT(0, 1, 32, 32, 0); PT(0, 2, 32, 32, 0); PT(0, 4, 32, 32, 0);
  // byte ratio
  PT(4, 1, 32, 32, 0); PT(2, 1, 32, 32, 0); PT(1, 1, 32, 32, 0); PT(2, 2, 32, 32, 0); PT(1, 2, 32, 32, 0); PT(1, 4, 32, 32, 0);
  PT(4, 1, 32, 32, 1); PT(2, 1, 32, 32, 1); PT(1, 1, 32, 32, 1); PT(2, 2, 32, 32, 1); PT(1, 2, 32, 32, 1); PT(1, 4, 32, 32, 1);
  // store shape
  PT(0, 2, 16, 16, 0); PT(2, 2, 16, 16, 0); PT(2, 2, 16, 16, 1);
  PT(0, 2, 16, 64, 0); PT(2, 2, 16, 64, 0); PT(2, 2, 16, 64, 1);
  PT(0, 2, 32, 64, 0); PT(2, 2, 32, 64, 0); PT(2, 2, 32, 64, 1);
  PT(0, 2, 4, 4, 0); PT(2, 2, 4, 4, 0); PT(2, 2, 4, 4, 1);
}

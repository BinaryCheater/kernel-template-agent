// What a store costs at the chip (r7 l2mix: 2 x 32 B per lane at 64 B apart 23015 cycles, 4 x 32 B
// contiguous 30761, 2 x 16 B contiguous 3765 = 1 x 32 B 3826: by the extent touched, not bytes): S
// stores per thread of W bytes per lane, lanes D bytes apart, 128 threads, grid G, warm. Reported:
// span on an SM (cycles). The l2mix pattern, stores only.
#include "../common.cuh"
constexpr int BLOCK = 4096 * 16;
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
template <int S, int W, int D>
__global__ void k(u64* out, u64* Y, unsigned* sink) {
  u64 c0 = clock64();
  u64 s = threadIdx.x;
  char* base = (char*)Y + (size_t)blockIdx.x * BLOCK;
#pragma unroll
  for (int i = 0; i < S; i++) store<W>((u64*)(base + i * 128 * D + threadIdx.x * D), s);
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
  u64 *o, *Y;
  unsigned* sink;
  const size_t B = 512ull * BLOCK + 4096;
  CK(cudaMalloc(&o, 8 * 3 * 512));
  CK(cudaMalloc(&Y, B));
  CK(cudaMalloc(&sink, 4));
  auto point = [&](auto kern, const char* name) {
    for (int nb : {128, 256, 512}) {
      std::vector<u64> h(3 * nb);
      Stat s = repeat([&] {
        CK(event_us([&] { kern<<<nb, 128>>>(o, Y, sink); }) < 0);
        CK(cudaMemcpy(h.data(), o, 8 * 3 * nb, cudaMemcpyDeviceToHost));
        return reduce(h, nb);
      });
      char p[48];
      snprintf(p, sizeof p, "%s_g%d", name, nb);
      emit("store_lines", p, s, "cycles");
    }
  };
#define PT(S, W, D) point(k<S, W, D>, "S" #S "_W" #W "_D" #D)
  PT(1, 32, 32); PT(2, 32, 32); PT(3, 32, 32); PT(4, 32, 32); PT(6, 32, 32); PT(8, 32, 32);
  PT(2, 16, 16); PT(4, 16, 16); PT(8, 16, 16);
  PT(2, 16, 32); PT(4, 16, 32); PT(1, 16, 64); PT(2, 16, 64); PT(4, 16, 64);
  PT(1, 32, 64); PT(2, 32, 64); PT(4, 32, 64);
  PT(4, 4, 4); PT(8, 4, 4); PT(2, 4, 32); PT(4, 4, 32); PT(2, 4, 64);
  PT(1, 32, 128); PT(2, 32, 128); PT(1, 4, 128); PT(2, 4, 128);
}

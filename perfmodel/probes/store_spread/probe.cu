// What lanes' spread costs at the chip's store limit (question 25; inventory §6 chip store limit:
// store_lines holds 420 B/clk of sectors for lanes up to 128 B apart, TileLang's tile store, 4 B per
// lane, 8 rows per instruction, keeps 244; HGEMM rotating Spearman 0.93 -> 0.85). 128 threads, 8
// stores per thread of 4 B per lane; LPR lanes contiguous per row, rows ROW bytes apart, each store
// STEP bytes along the row. Reported: span on an SM (cycles).
#include "../common.cuh"
typedef unsigned long long u64;
template <int LPR, int ROW, int STEP>
__global__ void __launch_bounds__(128) k(u64* out, unsigned* Y, unsigned* sink) {
  u64 c0 = clock64();
  const int t = threadIdx.x;
  unsigned s = t;
  char* base = (char*)Y + (size_t)blockIdx.x * 32 * ROW;
#pragma unroll
  for (int i = 0; i < 8; i++)
    *(unsigned*)(base + (t / LPR) * ROW + (t % LPR) * 4 + i * STEP) = s + i;
  __syncthreads();
  u64 c1 = clock64();
  if (t == 0) {
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
  u64* o;
  unsigned *Y, *sink;
  CK(cudaMalloc(&o, 8 * 3 * 512));
  CK(cudaMalloc(&Y, 512ull * 32 * 8192 + 8192));
  CK(cudaMalloc(&sink, 4));
  auto point = [&](auto kern, const char* name) {
    for (int nb : {256, 512}) {
      std::vector<u64> h(3 * nb);
      Stat s = repeat([&] {
        CK(event_us([&] { kern<<<nb, 128>>>(o, Y, sink); }) < 0);
        CK(cudaMemcpy(h.data(), o, 8 * 3 * nb, cudaMemcpyDeviceToHost));
        return reduce(h, nb);
      });
      char p[48];
      snprintf(p, sizeof p, "%s_g%d", name, nb);
      emit("store_spread", p, s, "cycles");
    }
  };
#define PT(L, R, S) point(k<L, R, S>, "L" #L "_R" #R "_S" #S)
  PT(4, 128, 16); PT(4, 256, 16); PT(4, 512, 16); PT(4, 1024, 16); PT(4, 2048, 16); PT(4, 4096, 16); PT(4, 8192, 16);
  PT(4, 4096, 32); PT(4, 4096, 8); PT(2, 4096, 8); PT(8, 4096, 32); PT(16, 4096, 64); PT(32, 4096, 128);
  PT(8, 512, 32); PT(16, 512, 64); PT(32, 512, 128);
}

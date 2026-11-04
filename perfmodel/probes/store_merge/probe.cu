// Which stores merge in the queue (question 40; D833: store_spread L2_R4096_S8 -59 %, rw_* G 768 +68
// to +112 %). store_spread's kernel: 128 threads, 8 stores per thread of 4 B per lane, LPR lanes
// contiguous per row, rows ROW bytes apart, each store STEP bytes along the row. STEP below 32 keeps
// a warp's next store in its own sectors. Reported: span on an SM (cycles); L2 write sectors by ncu.
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
    for (int nb : {128, 512}) {
      std::vector<u64> h(3 * nb);
      Stat s = repeat([&] {
        CK(event_us([&] { kern<<<nb, 128>>>(o, Y, sink); }) < 0);
        CK(cudaMemcpy(h.data(), o, 8 * 3 * nb, cudaMemcpyDeviceToHost));
        return reduce(h, nb);
      });
      char p[48];
      snprintf(p, sizeof p, "%s_g%d", name, nb);
      emit("store_merge", p, s, "cycles");
    }
  };
#define PT(L, R, S) point(k<L, R, S>, "L" #L "_R" #R "_S" #S)
  PT(1, 4096, 4); PT(1, 4096, 8); PT(1, 4096, 32);
  PT(2, 4096, 4); PT(2, 4096, 8); PT(2, 4096, 16); PT(2, 512, 8);
  PT(4, 4096, 4); PT(4, 4096, 8); PT(4, 512, 8);
  PT(8, 4096, 8); PT(8, 4096, 16); PT(8, 512, 16);
  PT(16, 4096, 16); PT(16, 4096, 32); PT(32, 4096, 32);
}

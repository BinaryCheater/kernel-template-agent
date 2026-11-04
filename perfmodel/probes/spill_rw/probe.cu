// Stack loads against stack stores (inventory §3 local memory; the spill probe of r7b had the
// compiler fold its loads: every stack instruction there was a store, so its 7.4 and 23 cycles per
// instruction are 13 and 45 per store; 256x256x32 t256 replays +32 %). N words of stack per thread,
// ST stores and LD loads per word, M passes, 256 threads; the loads cannot be forwarded (the index
// comes from a kernel argument). H > 0: 15 of 16 accesses stay in the first H words. G 1: one block;
// 48: one per SM. Reported: span on an SM (cycles).
#include "../common.cuh"
typedef unsigned long long u64;
constexpr int M = 4;
template <int T, int N, int LD, int ST, int H>
__global__ void __launch_bounds__(T) k(u64* out, unsigned* sink, unsigned z0, unsigned z1) {
  volatile unsigned a[N];
  unsigned s = threadIdx.x;
#pragma unroll 1
  for (int i = 0; i < N; i++) a[(i + z0) & (N - 1)] = s;
  __syncthreads();
  u64 c0 = clock64();
#pragma unroll 1
  for (int r = 0; r < M; r++) {
#pragma unroll 1
    for (int i = 0; i < N; i++) {
      int j = H && (i & 15) ? (i & (H - 1)) : i;
#pragma unroll
      for (int q = 0; q < ST; q++) a[(j + z0) & (N - 1)] = s + i;
#pragma unroll
      for (int q = 0; q < LD; q++) s ^= a[(j + z1) & (N - 1)];
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
  u64* o;
  unsigned* sink;
  CK(cudaMalloc(&o, 8 * 3 * 48));
  CK(cudaMalloc(&sink, 4));
  auto point = [&](auto kern, const char* name, int t) {
    for (int nb : {1, 48}) {
      std::vector<u64> h(3 * nb);
      Stat s = repeat([&] {
        CK(event_us([&] { kern<<<nb, t>>>(o, sink, 0u, 0u); }) < 0);
        CK(cudaMemcpy(h.data(), o, 8 * 3 * nb, cudaMemcpyDeviceToHost));
        return reduce(h, nb);
      });
      char p[48];
      snprintf(p, sizeof p, "%s_g%d", name, nb);
      emit("spill_rw", p, s, "cycles");
    }
  };
#define PT(N, L, S, H) point(k<256, N, L, S, H>, "n" #N "_L" #L "_S" #S "_H" #H, 256)
  PT(64, 1, 0, 0); PT(64, 0, 1, 0); PT(64, 1, 1, 0); PT(64, 4, 1, 0); PT(64, 1, 4, 0);
  PT(256, 1, 0, 0); PT(256, 0, 1, 0); PT(256, 1, 1, 0); PT(256, 4, 1, 0); PT(256, 1, 4, 0);
  PT(1024, 1, 0, 0); PT(1024, 0, 1, 0); PT(1024, 1, 1, 0); PT(1024, 4, 1, 0);
  PT(1024, 1, 1, 32); PT(1024, 1, 1, 128); PT(1024, 4, 1, 32); PT(1024, 4, 1, 128);
}

// 32-byte-per-lane global accesses, the width TileLang's row kernels use (`tl::load_global_256`,
// `tl::store_global_256`: LDG.E.ENL2.256, STG.E.ENL2.256; inventory §6 store port and chip store
// limit are measured at 4 and 16 B per lane only, and sm.py takes a 256-bit store as 128 B per warp
// instruction). A block of 128 threads does I accesses per thread, contiguous lanes, 1024 B per warp
// instruction, warm, 8 KiB per block as RMSNorm: stores alone, loads alone, both. Grids 1, 2, 16, 48,
// 96, 512 give the pair's port and the chip limit. Reported: the span on an SM (cycles), the launch
// by events.
#include "../common.cuh"
constexpr int I = 2;
template <int LD, int ST>
__global__ void k(unsigned long long* out, const unsigned long long* X, unsigned long long* Y, unsigned* sink) {
  unsigned long long c0 = clock64();
  unsigned long long a = 0, b = 0, c = 0, d = 0, s = threadIdx.x;
#pragma unroll
  for (int i = 0; i < I; i++) {
    const unsigned long long* p = X + blockIdx.x * 1024 + i * 512 + threadIdx.x * 4;
    if (LD) {
      asm volatile("ld.global.nc.L2::256B.v4.u64 {%0,%1,%2,%3}, [%4];"
                   : "=l"(a), "=l"(b), "=l"(c), "=l"(d)
                   : "l"(p));
      s ^= a ^ b ^ c ^ d;
    }
    if (ST)
      asm volatile("st.global.v4.u64 [%0], {%1,%2,%3,%4};"
                   :
                   : "l"(Y + (p - X)), "l"(s), "l"(s), "l"(s), "l"(s)
                   : "memory");
  }
  __syncthreads();
  unsigned long long c1 = clock64();
  if (threadIdx.x == 0) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    unsigned long long* o = out + 3 * blockIdx.x;
    o[0] = smid, o[1] = c0, o[2] = c1;
  }
  if (s == 12345) sink[0] = 1;
}
static double reduce(const std::vector<unsigned long long>& h, int nb) {
  std::vector<unsigned long long> lo(256, ~0ull), hi(256, 0);
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
  unsigned long long *o, *X, *Y;
  unsigned* sink;
  const size_t B = 512 * 8192 + 4096;
  CK(cudaMalloc(&o, 8 * 3 * 512));
  CK(cudaMalloc(&X, B));
  CK(cudaMalloc(&Y, B));
  CK(cudaMalloc(&sink, 4));
  CK(cudaMemset(X, 0, B));
  auto point = [&](auto kern, const char* name) {
    for (int nb : {1, 2, 16, 48, 96, 512}) {
      std::vector<unsigned long long> h(3 * nb);
      std::vector<double> ev;
      Stat s = repeat([&] {
        ev.push_back(event_us([&] { kern<<<nb, 128>>>(o, X, Y, sink); }));
        CK(cudaMemcpy(h.data(), o, 8 * 3 * nb, cudaMemcpyDeviceToHost));
        return reduce(h, nb);
      });
      ev.erase(ev.begin(), ev.begin() + WARM);
      char p[48], x[48];
      snprintf(x, sizeof x, "event_us=%.3f", stat_of(ev).median);
      snprintf(p, sizeof p, "%s_g%d", name, nb);
      emit("st256", p, s, "cycles", x);
    }
  };
  point(k<0, 1>, "st");
  point(k<1, 0>, "ld");
  point(k<1, 1>, "ld_st");
}

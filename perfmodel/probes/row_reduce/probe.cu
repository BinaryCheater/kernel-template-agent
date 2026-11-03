// Where a short block's time goes (STATE.md question 15: RMSNorm warm and softmax replayed 1.6-2.3x
// fast in a graph; inventory §1 "Block turnover", §2 shuffles and barriers). A block of 128 threads
// does the row kernels' steps: LD two 32-byte global loads per thread (warm rows, stride 8 KiB per
// block), RED the row reduction (five shuffles per warp, the warp's sum through shared memory, one
// block barrier, re-read), ST two 32-byte global stores per thread. Grid 48 = one block per SM; grid
// 512 = 11 blocks per SM in turn, as the test kernels. Thread 0 records %smid and clock64 at entry and
// exit; reported per point: span (first entry to last exit on an SM), block (entry to exit), gap
// (exit to the next entry on that SM), medians over SMs.
#include "../common.cuh"
constexpr int RED_BYTES = 512;
template <int LD, int RED, int ST>
__global__ void k(unsigned long long* out, const float* X, float* Y, unsigned* sink) {
  extern __shared__ __align__(128) float ws[];
  unsigned long long c0 = clock64();
  float4 a = {0.f, 0.f, 0.f, 0.f}, b = a;
  const int t = threadIdx.x, off = blockIdx.x * 2048 + t * 16;
  if (LD) {
    a = *(const float4*)(X + off);
    b = *(const float4*)(X + off + 8);
  }
  float v = a.x * a.x + a.y * a.y + b.x * b.x + b.y * b.y;
  if (RED) {
#pragma unroll
    for (int u = 0; u < 5; u++) v += __shfl_xor_sync(0xffffffffu, v, 1 << u);
    const int w = t / 32;
    if ((t & 31) == 0) ws[w] = v;
    __syncthreads();
    v = ws[0] + ws[1] + ws[2] + ws[3];
  }
  if (ST) {
    float4 r = {a.x * v, a.y * v, a.z * v, a.w * v};
    *(float4*)(Y + off) = r;
    r = {b.x * v, b.y * v, b.z * v, b.w * v};
    *(float4*)(Y + off + 8) = r;
  }
  __syncthreads();
  unsigned long long c1 = clock64();
  if (t == 0) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    unsigned long long* o = out + 3 * blockIdx.x;
    o[0] = smid, o[1] = c0, o[2] = c1;
  }
  if (v == 12345.f) sink[0] = 1;
}
struct Res {
  double span, block, gap;
};
static Res reduce(const std::vector<unsigned long long>& h, int nb) {
  std::vector<std::vector<std::pair<unsigned long long, unsigned long long>>> sm(256);
  for (int b = 0; b < nb; b++) sm[h[3 * b] % 256].push_back({h[3 * b + 1], h[3 * b + 2]});
  std::vector<double> span, block, gap;
  for (auto& s : sm) {
    if (s.empty()) continue;
    std::sort(s.begin(), s.end());
    unsigned long long x1 = 0;
    for (size_t i = 0; i < s.size(); i++) {
      x1 = std::max(x1, s[i].second);
      block.push_back((double)(s[i].second - s[i].first));
      if (i) gap.push_back((double)s[i].first - (double)s[i - 1].second);
    }
    span.push_back((double)(x1 - s[0].first));
  }
  return {stat_of(span).median, stat_of(block).median, gap.empty() ? 0 : stat_of(gap).median};
}
int main() {
  unsigned long long* o;
  float *X, *Y;
  unsigned* sink;
  const size_t B = 512 * 8192 + 4096;
  CK(cudaMalloc(&o, 8 * 3 * 512));
  CK(cudaMalloc(&X, B));
  CK(cudaMalloc(&Y, B));
  CK(cudaMalloc(&sink, 4));
  CK(cudaMemset(X, 0, B));
  auto point = [&](auto kern, const char* name) {
    CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, RED_BYTES));
    for (int nb : {48, 512}) {
      std::vector<unsigned long long> h(3 * nb);
      std::vector<double> bl, gp, ev;
      Stat s = repeat([&] {
        ev.push_back(event_us([&] { kern<<<nb, 128, RED_BYTES>>>(o, X, Y, sink); }));
        CK(cudaMemcpy(h.data(), o, 8 * 3 * nb, cudaMemcpyDeviceToHost));
        Res r = reduce(h, nb);
        bl.push_back(r.block);
        gp.push_back(r.gap);
        return r.span;
      });
      bl.erase(bl.begin(), bl.begin() + WARM);
      gp.erase(gp.begin(), gp.begin() + WARM);
      ev.erase(ev.begin(), ev.begin() + WARM);
      char p[48], x[48];
      snprintf(x, sizeof x, "event_us=%.3f", stat_of(ev).median);
      snprintf(p, sizeof p, "%s_g%d_span", name, nb);
      emit("row_reduce", p, s, "cycles", x);
      snprintf(p, sizeof p, "%s_g%d_block", name, nb);
      emit("row_reduce", p, stat_of(bl), "cycles", x);
      if (nb > 48) {
        snprintf(p, sizeof p, "%s_g%d_gap", name, nb);
        emit("row_reduce", p, stat_of(gp), "cycles", x);
      }
    }
  };
  point(k<0, 0, 0>, "empty");
  point(k<1, 0, 0>, "ld");
  point(k<0, 1, 0>, "red");
  point(k<0, 0, 1>, "st");
  point(k<1, 0, 1>, "ld_st");
  point(k<1, 1, 0>, "ld_red");
  point(k<1, 1, 1>, "ld_red_st");
}

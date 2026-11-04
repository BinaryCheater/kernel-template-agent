// cp.async from DRAM (questions 10, 28; inventory §5 DRAM latency, DRAM queue per SM; HGEMM s1
// rotating: replay +18.6 %, measured -1.8 %). 128 threads, 8 copies of 16 B per thread per step, all
// contiguous, one stage: every step waits for its own copies, then Z passes of an FFMA loop, then a
// block barrier. A launch reads G x 16 steps once. rot: the start walks a 224 MiB window (cold);
// warm: the same bytes every launch. Reported: span on an SM (cycles).
#include "../common.cuh"
typedef unsigned long long u64;
constexpr int T = 128, C = 8, STEPS = 16, STEP_BYTES = C * T * 16;
__device__ __forceinline__ void cp16(unsigned dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
template <int Z>
__global__ void __launch_bounds__(T, 1) k(u64* out, const char* src, float* sink) {
  extern __shared__ __align__(128) unsigned char buf[];
  const unsigned sm = (unsigned)__cvta_generic_to_shared(buf);
  const int t = threadIdx.x;
  const char* p = src + (size_t)blockIdx.x * STEPS * STEP_BYTES + t * 16;
  float f = t;
  u64 c0 = clock64();
#pragma unroll 1
  for (int s = 0; s < STEPS; s++) {
#pragma unroll
    for (int c = 0; c < C; c++) cp16(sm + (c * T + t) * 16, p + s * STEP_BYTES + c * T * 16);
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#pragma unroll 1
    for (int z = 0; z < Z; z++) f = fmaf(f, 0.9999f, 0.25f);
    __syncthreads();
  }
  u64 c1 = clock64();
  if (t == 0) {
    unsigned smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    u64* o = out + 3 * blockIdx.x;
    o[0] = smid, o[1] = c0, o[2] = c1;
  }
  if (f == 1.2345f) sink[0] = f;
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
  char* src;
  float* sink;
  const size_t W = 224ull << 20;
  CK(cudaMalloc(&o, 8 * 3 * 48));
  CK(cudaMalloc(&src, W + (64 << 20)));
  CK(cudaMalloc(&sink, 4));
  CK(cudaMemset(src, 1, W + (64 << 20)));
  auto point = [&](auto kern, int z, int nb, bool rot) {
    std::vector<u64> h(3 * nb);
    size_t off = 0, len = (size_t)nb * STEPS * STEP_BYTES;
    Stat s = repeat([&] {
      CK(event_us([&] { kern<<<nb, T, STEP_BYTES>>>(o, src + off, sink); }) < 0);
      if (rot) off = (off + len + (1 << 20)) % W;
      CK(cudaMemcpy(h.data(), o, 8 * 3 * nb, cudaMemcpyDeviceToHost));
      return reduce(h, nb);
    });
    char p[48];
    snprintf(p, sizeof p, "%s_z%d_g%d", rot ? "rot" : "warm", z, nb);
    emit("copy_dram", p, s, "cycles");
  };
  for (int rot : {0, 1})
    for (int nb : {1, 16, 48}) {
      point(k<0>, 0, nb, rot);
      point(k<64>, 64, nb, rot);
      point(k<256>, 256, nb, rot);
      point(k<1024>, 1024, nb, rot);
    }
}

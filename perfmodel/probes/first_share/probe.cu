// Misses of shared tiles at the chip's scale (questions 8, 30, 32; HGEMM rotating over warm is +17
// to +22 us at 6 waves of blocks, +4 to +6 at 3; s1 rotating replay +18 %, measured +1 %).
// share: 48 blocks, 8 copies of 16 B per thread per step, one stage; D blocks read each tile at the
// same time, Z FFMA passes between the steps. seq: 96 x w blocks at two blocks per SM (98304 B of
// shared memory each, the driver refuses 120 KB), 4 blocks to a tile, 24 tiles: the later waves read
// what the first left in the L2. pieces: one block, each thread reads S bytes further along its own
// line per step: how many lines a miss brings. rot: a 45 MB read between launches.
// Reported: span on an SM (share), a wave's median block (seq), cycles per step (pieces).
#include "../common.cuh"
typedef unsigned long long u64;
constexpr int T = 128, C = 8, STEPS = 16, STEP_BYTES = C * T * 16, SEQ_SMEM = 98304;
constexpr int PSTEPS = 64;
__device__ __forceinline__ void cp16(unsigned dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cpwait() {
  asm volatile("cp.async.commit_group;" ::: "memory");
  asm volatile("cp.async.wait_group 0;" ::: "memory");
}
// a read of every third line of 45 MB between timed launches: what a rotating launch reads is out of the L2
__global__ void __launch_bounds__(1024, 1) flush(const char* p, unsigned* sink) {
  const size_t i = (size_t)(blockIdx.x * 1024 + threadIdx.x) * 384;
  unsigned s = 0;
  for (int k = 0; k < 3; k++) {
    unsigned v;
    asm volatile("ld.global.u32 %0, [%1];" : "=r"(v) : "l"(p + i + k * 128) : "memory");
    s += v;
  }
  if (s == 0x12345) sink[0] = s;
}
template <int Z, int D, int MOD>
__global__ void __launch_bounds__(T, 1) share(u64* out, const char* src, float* sink) {
  extern __shared__ __align__(128) unsigned char buf[];
  const unsigned sm = (unsigned)__cvta_generic_to_shared(buf);
  const int t = threadIdx.x;
  const int tile = (MOD ? blockIdx.x % MOD : blockIdx.x) / D;
  const char* p = src + (size_t)tile * STEPS * STEP_BYTES + t * 16;
  float f = t;
  u64 c0 = clock64();
#pragma unroll 1
  for (int s = 0; s < STEPS; s++) {
#pragma unroll
    for (int c = 0; c < C; c++) cp16(sm + (c * T + t) * 16, p + s * STEP_BYTES + c * T * 16);
    cpwait();
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
template <int S>
__global__ void __launch_bounds__(T, 1) pieces(u64* out, const char* src, float* sink) {
  extern __shared__ __align__(128) unsigned char buf[];
  const unsigned sm = (unsigned)__cvta_generic_to_shared(buf);
  const int t = threadIdx.x;
  const char* p = src + t * 8192;
  float f = t;
  u64 c0 = clock64();
#pragma unroll 1
  for (int s = 0; s < PSTEPS; s++) {
    cp16(sm + t * 16, p + s * S);
    cpwait();
#pragma unroll 1
    for (int z = 0; z < 256; z++) f = fmaf(f, 0.9999f, 0.25f);
    __syncthreads();
  }
  u64 c1 = clock64();
  if (t == 0) out[0] = c1 - c0;
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
static double wave_median(const std::vector<u64>& h, int from, int to) {
  std::vector<double> v;
  for (int b = from; b < to; b++) v.push_back((double)(h[3 * b + 2] - h[3 * b + 1]));
  return stat_of(v).median;
}
int main() {
  u64* o;
  char* src;
  char* other;
  float* sink;
  const size_t W = 16 << 20, F = 48 << 20;
  CK(cudaMalloc(&o, 8 * 3 * 384));
  CK(cudaMalloc(&src, W + (1 << 20)));
  CK(cudaMalloc(&other, F));
  CK(cudaMalloc(&sink, 4));
  CK(cudaMemset(src, 1, W + (1 << 20)));
  CK(cudaMemset(other, 1, F));
  auto share_pt = [&](auto kern, int z, int d, bool rot) {
    const int nb = 48;
    std::vector<u64> h(3 * nb);
    Stat s = repeat([&] {
      if (rot) flush<<<112, 1024>>>(other, (unsigned*)sink);
      kern<<<nb, T, STEP_BYTES>>>(o, src, sink);
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(h.data(), o, 8 * 3 * nb, cudaMemcpyDeviceToHost));
      return reduce(h, nb);
    });
    char p[48];
    snprintf(p, sizeof p, "%s_share_d%d_z%d", rot ? "rot" : "warm", d, z);
    emit("first_share", p, s, "cycles");
  };
  auto seq_pt = [&](auto kern, int waves, bool rot, const char* tag = "") {
    const int nb = 96 * waves;
    CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, SEQ_SMEM));
    std::vector<u64> h(3 * nb);
    std::vector<double> first, last;
    Stat s = repeat([&] {
      if (rot) flush<<<112, 1024>>>(other, (unsigned*)sink);
      kern<<<nb, T, SEQ_SMEM>>>(o, src, sink);
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(h.data(), o, 8 * 3 * nb, cudaMemcpyDeviceToHost));
      first.push_back(wave_median(h, 0, 96));
      last.push_back(wave_median(h, nb - 96, nb));
      return first.back();
    });
    char p[48];
    snprintf(p, sizeof p, "%s_seq%s_w%d_first", rot ? "rot" : "warm", tag, waves);
    emit("first_share", p, s, "cycles");
    snprintf(p, sizeof p, "%s_seq%s_w%d_last", rot ? "rot" : "warm", tag, waves);
    emit("first_share", p, stat_of(last), "cycles");
  };
  auto piece_pt = [&](auto kern, int stride, bool rot) {
    Stat s = repeat([&] {
      if (rot) flush<<<112, 1024>>>(other, (unsigned*)sink);
      kern<<<1, T, 2048>>>(o, src, sink);
      CK(cudaDeviceSynchronize());
      u64 h;
      CK(cudaMemcpy(&h, o, 8, cudaMemcpyDeviceToHost));
      return (double)h / PSTEPS;
    });
    char p[48];
    snprintf(p, sizeof p, "%s_pieces_s%d", rot ? "rot" : "warm", stride);
    emit("first_share", p, s, "cycles");
  };
  for (int rot : {0, 1}) {
    share_pt(share<0, 1, 0>, 0, 1, rot);
    share_pt(share<0, 2, 0>, 0, 2, rot);
    share_pt(share<0, 4, 0>, 0, 4, rot);
    share_pt(share<0, 8, 0>, 0, 8, rot);
    share_pt(share<0, 16, 0>, 0, 16, rot);
    share_pt(share<0, 48, 0>, 0, 48, rot);
    share_pt(share<256, 1, 0>, 256, 1, rot);
    share_pt(share<256, 4, 0>, 256, 4, rot);
    share_pt(share<256, 16, 0>, 256, 16, rot);
    share_pt(share<256, 48, 0>, 256, 48, rot);
    seq_pt(share<0, 4, 96>, 2, rot);
    seq_pt(share<0, 4, 96>, 4, rot);
    seq_pt(share<0, 2, 96>, 2, rot, "d2");
    piece_pt(pieces<16>, 16, rot);
    piece_pt(pieces<32>, 32, rot);
    piece_pt(pieces<64>, 64, rot);
    piece_pt(pieces<128>, 128, rot);
  }
}

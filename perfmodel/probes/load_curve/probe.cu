// L2 latency against offered load (LOOP contention group; inventory §5, questions 32 and the
// what-if magnitudes). G blocks of 4 warps, warm 4 MiB tile. Warp 0 of every block times a lone
// 512-byte cp.async (commit, wait 0) every 32 FFMA passes; warps 1-3 are the load: one 512-byte
// cp.async per pass with up to one group in flight, then Z FFMA passes.
// Reported: warp 0's cycles per copy (block 0), and the launch's span by events.
#include "../common.cuh"
typedef unsigned long long u64;
constexpr int T = 128, STEPS = 32, MARK_GAP = 32;
constexpr size_t TILE = 4 << 20;
__device__ __forceinline__ void cp16(unsigned dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void wait0() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }
__device__ __forceinline__ void wait1() { asm volatile("cp.async.wait_group 1;" ::: "memory"); }
template <int Z>
__global__ void __launch_bounds__(T, 1) k(u64* out, const char* src, float* sink) {
  extern __shared__ __align__(128) unsigned char buf[];
  const unsigned sm = (unsigned)__cvta_generic_to_shared(buf);
  const int t = threadIdx.x, w = t >> 5, lane = t & 31, b = blockIdx.x;
  float f = t;
  u64 acc = 0;
  const unsigned base = (unsigned)(b * 1033) * 512;
  if (w == 0) {
#pragma unroll 1
    for (int s = 0; s < STEPS; s++) {
#pragma unroll 1
      for (int z = 0; z < MARK_GAP; z++) f = fmaf(f, 0.9999f, 0.25f);
      u64 c0 = clock64();
      cp16(sm + lane * 16, src + (base + s * 3584) % TILE + lane * 16);
      commit();
      wait0();
      acc += clock64() - c0;
    }
    if (t == 0 && b == 0) out[0] = acc;
  } else {
    constexpr int PASSES = 40000 / (400 + Z);
#pragma unroll 1
    for (int s = 0; s < PASSES; s++) {
      cp16(sm + 512 * w + lane * 16, src + (base + (s * 3 + w * 5000) * 512) % TILE + lane * 16);
      commit();
      wait1();
#pragma unroll 1
      for (int z = 0; z < Z; z++) f = fmaf(f, 0.9999f, 0.25f);
    }
    wait0();
  }
  if (f == 1.2345f) sink[0] = f;
}
int main() {
  u64* o;
  char* src;
  float* sink;
  CK(cudaMalloc(&o, 8));
  CK(cudaMalloc(&src, TILE));
  CK(cudaMalloc(&sink, 4));
  CK(cudaMemset(src, 1, TILE));
  auto point = [&](auto kern, int z, int g) {
    double span = 0;
    Stat s = repeat([&] {
      span = event_us([&] { kern<<<g, T, 4096>>>(o, src, sink); });
      u64 h;
      CK(cudaMemcpy(&h, o, 8, cudaMemcpyDeviceToHost));
      return (double)h / STEPS;
    });
    char p[48];
    snprintf(p, sizeof p, "g%d_z%d", g, z);
    char x[48];
    snprintf(x, sizeof x, "span_us=%.2f", span);
    emit("load_curve", p, s, "cycles", x);
  };
  for (int g : {1, 12, 24, 48, 96, 144}) {
#define PT(Z) point(k<Z>, Z, g);
    PT(0) PT(16) PT(64) PT(256) PT(1024)
  }
}

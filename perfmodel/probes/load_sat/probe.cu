// L2 latency at the saturated chip (r10b: load_curve reached a tenth to a third of the L2 rate). As
// load_curve, the load warps issue 16 copies per pass, up to one group in flight, then Z passes.
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
    constexpr int PASSES = 60000 / (1000 + Z);
#pragma unroll 1
    for (int s = 0; s < PASSES; s++) {
#pragma unroll 1
      for (int c = 0; c < 16; c++)
        cp16(sm + 512 * (w + (c & 3)) + lane * 16,
             src + (base + (s * 48 + c * 3 + w * 5000) * 512) % TILE + lane * 16);
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
    emit("load_sat", p, s, "cycles", x);
  };
  for (int g : {12, 24, 48, 96, 144}) {
#define PT(Z) point(k<Z>, Z, g);
    PT(0) PT(64) PT(256)
  }
}

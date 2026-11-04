// The chip's L2 read rate by pattern (question 37; r11): every warp of G blocks reads a warm 8 MiB
// without a pause, by plain loads or copies, lanes S bytes apart. load_sat moved about 1900 B/clk
// of contiguous copies against the frozen 817 for TileLang tiles; copy_depth 42 B/clk on one SM
// against the port's 30.7. Reported: the launch's span. Byte and line rates follow from it.
#include "../common.cuh"
typedef unsigned long long u64;
constexpr int T = 256, PASSES = 128;
constexpr size_t F = 8 << 20;
__device__ __forceinline__ void cp16(unsigned dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void wait1() { asm volatile("cp.async.wait_group 1;" ::: "memory"); }
__device__ __forceinline__ void wait0() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }
__device__ __forceinline__ unsigned off(int b, int w, int s) {
  return (unsigned)(b * 1033 + w * 4099 + s * 48) * 512;
}
template <int S>
__global__ void __launch_bounds__(T, 1) ld(const char* src, unsigned* sink) {
  const int t = threadIdx.x, w = t >> 5, lane = t & 31, b = blockIdx.x;
  unsigned acc = 0;
#pragma unroll 1
  for (int s = 0; s < PASSES; s++)
    acc ^= *(const unsigned*)(src + ((off(b, w, s) + lane * S) & (unsigned)(F - 1)));
  if (acc == 0x12345) sink[0] = acc;
}
template <int S>
__global__ void __launch_bounds__(T, 1) cp(const char* src, unsigned* sink) {
  extern __shared__ __align__(128) unsigned char buf[];
  const unsigned sm = (unsigned)__cvta_generic_to_shared(buf);
  const int t = threadIdx.x, w = t >> 5, lane = t & 31, b = blockIdx.x;
#pragma unroll 1
  for (int s = 0; s < PASSES / 4; s++) {
#pragma unroll 1
    for (int c = 0; c < 4; c++)
      cp16(sm + 512 * (w + (c & 3)) + lane * 16,
           src + ((off(b, w, s * 4 + c) + lane * S) & (unsigned)(F - 1)));
    commit();
    wait1();
  }
  wait0();
}
int main() {
  char* src;
  unsigned* sink;
  CK(cudaMalloc(&src, F));
  CK(cudaMalloc(&sink, 4));
  CK(cudaMemset(src, 1, F));
  auto point = [&](auto kern, const char* name, int g, int smem) {
    Stat s = repeat([&] {
      return event_us([&] { kern<<<g, T, smem>>>(src, sink); });
    });
    char p[48];
    snprintf(p, sizeof p, "%s_g%d", name, g);
    emit("l2_cap", p, s, "us");
  };
  for (int g : {12, 48, 144, 480}) {
    point(ld<4>, "ld4", g, 0);
    point(ld<32>, "ld32", g, 0);
    point(ld<128>, "ld128", g, 0);
    point(cp<16>, "cp16", g, 8192);
    point(cp<64>, "cp64", g, 8192);
    point(cp<128>, "cp128", g, 8192);
  }
}

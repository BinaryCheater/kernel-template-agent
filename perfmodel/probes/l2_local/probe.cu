// What separates TileLang's 817 B/clk from l2_cap's 2100 for contiguous copies (question 37): the
// same 512-byte copies from a warm 8 MiB, the addresses arranged differently. MODE 0 scattered as
// l2_cap; 1 each block walks its own region in order; 2 every block reads the same addresses in
// lockstep; 3 each warp stays in a 4 KiB page for a pass, pages scattered. Reported: the span.
#include "../common.cuh"
constexpr int T = 256, PASSES = 32;
constexpr size_t F = 8 << 20;
__device__ __forceinline__ void cp16(unsigned dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
__device__ __forceinline__ void wait1() { asm volatile("cp.async.wait_group 1;" ::: "memory"); }
__device__ __forceinline__ void wait0() { asm volatile("cp.async.wait_group 0;" ::: "memory"); }
template <int MODE>
__device__ __forceinline__ unsigned off(int b, int w, int s, int c) {
  if constexpr (MODE == 0) return (unsigned)(b * 1033 + w * 4099 + (s * 4 + c) * 48) * 512;
  if constexpr (MODE == 1) return (unsigned)(b << 17) + (unsigned)((s * 4 + c) * 8 + w) * 512;
  if constexpr (MODE == 2) return (unsigned)((s * 4 + c) * 8 + w) * 512;
  return (unsigned)(b * 1033 + w * 4099 + s * 48) * 4096 + (unsigned)c * 512;
}
template <int MODE>
__global__ void __launch_bounds__(T, 1) cp(const char* src, unsigned* sink) {
  extern __shared__ __align__(128) unsigned char buf[];
  const unsigned sm = (unsigned)__cvta_generic_to_shared(buf);
  const int t = threadIdx.x, w = t >> 5, lane = t & 31, b = blockIdx.x;
#pragma unroll 1
  for (int s = 0; s < PASSES; s++) {
#pragma unroll 1
    for (int c = 0; c < 4; c++)
      cp16(sm + 512 * (w + (c & 3)) + lane * 16,
           src + ((off<MODE>(b, w, s, c) + lane * 16) & (unsigned)(F - 1)));
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
  auto point = [&](auto kern, const char* name, int g) {
    Stat s = repeat([&] { return event_us([&] { kern<<<g, T, 8192>>>(src, sink); }); });
    char p[48];
    snprintf(p, sizeof p, "%s_g%d", name, g);
    emit("l2_local", p, s, "us");
  };
  for (int g : {48, 144, 480}) {
    point(cp<0>, "scatter", g);
    point(cp<1>, "walk", g);
    point(cp<2>, "lockstep", g);
    point(cp<3>, "page", g);
  }
}

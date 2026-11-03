// Plain shared-memory loads and stores (inventory §3 "Plain shared loads/stores LDS, STS": absent).
// Pattern copied from upstream agentic/tilelang/microbench/smem_bw.cu: every address adds
// (previous result & zero), zero = 0 at run time, so nothing is dropped, merged or hoisted; with one
// warp the loads form one dependent chain (latency), with 4-32 warps on one SM the array rate shows.
// Kinds: LDS.32 / LDS.64 / LDS.128 (each lane 4 / 8 / 16 contiguous bytes, a warp 128 / 256 / 512 B,
// conflict-free), STS.32 / STS.128. Reported: cycles per warp instruction per SM (clock64 of warp 0
// around the loop, over ITERS x 8 x warps instructions). Round 2 adds 2, 3, 5 and 6 warps (STS.128 at
// 4 warps, STATE.md open question 4: a per-sub-partition limit predicts 5 warps slower than 4).
#include "../common.cuh"
constexpr int ITERS = 256;
template <int KIND>
__global__ void k(unsigned zero, unsigned long long* out, unsigned* sink) {
  extern __shared__ __align__(128) unsigned char buf[];
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  constexpr int W = (KIND == 0 || KIND == 3) ? 4 : (KIND == 1 ? 8 : 16);
  unsigned a = (unsigned)__cvta_generic_to_shared(buf) + (warp % 4) * 8192 + lane * W;
  unsigned r0 = lane, r1 = 1, r2 = 2, r3 = 3;
  __syncthreads();
  long long t0 = clock64();
#pragma unroll 1
  for (int i = 0; i < ITERS; i++) {
#pragma unroll
    for (int u = 0; u < 8; u++) {
      unsigned off = (u & 3) * 512 + (r0 & zero), x0, x1, x2, x3;
      if (KIND == 0) { asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x0) : "r"(a + off) : "memory"); r0 ^= x0; }
      if (KIND == 1) { asm volatile("ld.shared.v2.u32 {%0,%1}, [%2];" : "=r"(x0), "=r"(x1) : "r"(a + off) : "memory"); r0 ^= x0; r1 ^= x1; }
      if (KIND == 2) { asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(x0), "=r"(x1), "=r"(x2), "=r"(x3) : "r"(a + off) : "memory"); r0 ^= x0; r1 ^= x1; r2 ^= x2; r3 ^= x3; }
      if (KIND == 3) { asm volatile("st.shared.u32 [%0], %1;" ::"r"(a + off), "r"(r0 + u) : "memory"); }
      if (KIND == 4) { asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};" ::"r"(a + off), "r"(r0 + u), "r"(r1), "r"(r2), "r"(r3) : "memory"); }
    }
  }
  __syncthreads();  // every warp done: the block's time, not warp 0's
  long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  if ((r0 ^ r1 ^ r2 ^ r3) == 12345u) sink[0] = r0;  // every result live: nothing folds
}
int main() {
  unsigned long long* o;
  unsigned* sink;
  CK(cudaMalloc(&o, 8));
  CK(cudaMalloc(&sink, 4));
  const char* names[5] = {"LDS.32", "LDS.64", "LDS.128", "STS.32", "STS.128"};
  void (*fs[5])(unsigned, unsigned long long*, unsigned*) = {k<0>, k<1>, k<2>, k<3>, k<4>};
  for (int f = 0; f < 5; f++) CK(cudaFuncSetAttribute(fs[f], cudaFuncAttributeMaxDynamicSharedMemorySize, 40 * 1024));
  for (int f = 0; f < 5; f++)
    for (int warps : {1, 2, 3, 4, 5, 6, 8, 16, 32}) {
      unsigned long long h;
      Stat s = repeat([&] {
        fs[f]<<<1, 32 * warps, 40 * 1024>>>(0u, o, sink);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(&h, o, 8, cudaMemcpyDeviceToHost));
        return (double)h / (ITERS * 8.0 * warps);
      });
      char p[32];
      snprintf(p, sizeof p, "%s_w%d", names[f], warps);
      emit("smem_ldsts", p, s, "cycles/instr/SM");
    }
}

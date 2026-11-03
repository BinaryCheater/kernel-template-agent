// Narrow and strided shared-memory loads and stores (round 6; STATE.md question 21: w4gemm's LDS.U8
// and LDS.U16, 13-17 % of its path, against round 1's contiguous 4-16 B points; inventory §3 "Bank
// conflicts": none modelled for LDS/STS). Pattern of perfmodel/probes/smem_ldsts: every address adds
// (previous result & zero); at one warp the loads form one dependent chain (latency). Kinds: LDS.U8,
// LDS.U16, LDS.32, STS.U16, STS.32; lanes `stride` bytes apart (1 or 2 = the element size,
// contiguous; 16, 64); 1, 4 and 32 warps on one SM. Reported: cycles per warp instruction per SM.
#include "../common.cuh"
constexpr int ITERS = 256;
template <int KIND>
__global__ void k(unsigned stride, unsigned zero, unsigned long long* out, unsigned* sink) {
  extern __shared__ __align__(128) unsigned char buf[];
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  unsigned a = (unsigned)__cvta_generic_to_shared(buf) + (warp % 4) * 8192 + lane * stride;
  unsigned r0 = lane;
  __syncthreads();
  long long t0 = clock64();
#pragma unroll 1
  for (int i = 0; i < ITERS; i++) {
#pragma unroll
    for (int u = 0; u < 8; u++) {
      unsigned off = (u & 3) * 2048 + (r0 & zero), x0;
      if constexpr (KIND == 0) { asm volatile("ld.shared.u8 %0, [%1];" : "=r"(x0) : "r"(a + off) : "memory"); r0 ^= x0; }
      if constexpr (KIND == 1) { asm volatile("ld.shared.u16 %0, [%1];" : "=r"(x0) : "r"(a + off) : "memory"); r0 ^= x0; }
      if constexpr (KIND == 2) { asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x0) : "r"(a + off) : "memory"); r0 ^= x0; }
      if constexpr (KIND == 3) { asm volatile("st.shared.u16 [%0], %1;" ::"r"(a + off), "r"(r0 + u) : "memory"); }
      if constexpr (KIND == 4) { asm volatile("st.shared.u32 [%0], %1;" ::"r"(a + off), "r"(r0 + u) : "memory"); }
    }
  }
  __syncthreads();
  long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  if (r0 == 12345u) sink[0] = r0;
}
int main() {
  unsigned long long* o;
  unsigned* sink;
  CK(cudaMalloc(&o, 8));
  CK(cudaMalloc(&sink, 4));
  const char* names[5] = {"LDS.U8", "LDS.U16", "LDS.32", "STS.U16", "STS.32"};
  const unsigned elem[5] = {1, 2, 4, 2, 4};
  void (*fs[5])(unsigned, unsigned, unsigned long long*, unsigned*) = {k<0>, k<1>, k<2>, k<3>, k<4>};
  for (int f = 0; f < 5; f++) CK(cudaFuncSetAttribute(fs[f], cudaFuncAttributeMaxDynamicSharedMemorySize, 40 * 1024));
  for (int f = 0; f < 5; f++)
    for (unsigned st : {elem[f], 16u, 64u})
      for (int warps : {1, 4, 32}) {
        unsigned long long h;
        Stat s = repeat([&] {
          fs[f]<<<1, 32 * warps, 40 * 1024>>>(st, 0u, o, sink);
          CK(cudaDeviceSynchronize());
          CK(cudaMemcpy(&h, o, 8, cudaMemcpyDeviceToHost));
          return (double)h / (ITERS * 8.0 * warps);
        });
        char p[32];
        snprintf(p, sizeof p, "%s_s%u_w%d", names[f], st, warps);
        emit("lds_narrow", p, s, "cycles/instr/SM");
      }
}

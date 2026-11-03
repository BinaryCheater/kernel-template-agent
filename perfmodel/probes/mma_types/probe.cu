// Tensor pipe per opcode and data type (inventory §2 "Tensor pipe per sub-partition"; STATE.md: FP8
// rows replay QMMA at the HMMA hold). Each warp runs NA independent mma per iteration on register
// operands, no loads: NA = 16 on one warp and on four warps (one per sub-partition) gives cycles per
// instruction per sub-partition (the hold); NA = 1 chains the accumulator (result latency, or the
// hold where that is longer). Types: f16/bf16 k16 with f32 and f16 accumulators, e4m3/e5m2 k32 with
// f32 and f16, s8 k32, s4 k64. Not here because sm_121a's ptxas refuses them: bf16 accumulators and
// the block-scaled kind::mxf4nvf4 (NVFP4) mma.
#include "../common.cuh"
constexpr int ITERS = 256;
#define OPS4 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
#define OPS2 "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
#define IN4(x) "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1])
#define F4(s) asm volatile(s OPS4 : "+f"(f[0]), "+f"(f[1]), "+f"(f[2]), "+f"(f[3]) : IN4())
#define I4(s) asm volatile(s OPS4 : "+r"(u[0]), "+r"(u[1]), "+r"(u[2]), "+r"(u[3]) : IN4())
#define I2(s) asm volatile(s OPS2 : "+r"(u[0]), "+r"(u[1]) : IN4())
template <int T>
__device__ __forceinline__ void mma(float* f, unsigned* u, const unsigned* a, const unsigned* b) {
  if constexpr (T == 0) F4("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 ");
  if constexpr (T == 1) I2("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 ");
  if constexpr (T == 2) F4("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 ");
  if constexpr (T == 4) F4("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 ");
  if constexpr (T == 5) I2("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 ");
  if constexpr (T == 6) F4("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 ");
  if constexpr (T == 7) I4("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32.satfinite ");
  if constexpr (T == 8) I4("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32.satfinite ");
}
template <int T, int NA>
__global__ void k(unsigned z, unsigned long long* out, unsigned* sink) {
  float f[NA][4];
  unsigned u[NA][4];
  unsigned a[4], b[2];
#pragma unroll
  for (int i = 0; i < 4; i++) a[i] = threadIdx.x * (i + 1) + z;
#pragma unroll
  for (int i = 0; i < 2; i++) b[i] = threadIdx.x * (i + 5) + z;
#pragma unroll
  for (int c = 0; c < NA; c++)
#pragma unroll
    for (int i = 0; i < 4; i++) f[c][i] = 0.f, u[c][i] = 0;
  __syncthreads();
  long long t0 = clock64();
#pragma unroll 1
  for (int it = 0; it < ITERS; it++) {
#pragma unroll
    for (int c = 0; c < NA; c++) mma<T>(f[c], u[c], a, b);
  }
  __syncthreads();
  long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  unsigned s = 0;
#pragma unroll
  for (int c = 0; c < NA; c++)
#pragma unroll
    for (int i = 0; i < 4; i++) s ^= u[c][i] ^ (unsigned)f[c][i];
  if (s == 0x12345u) sink[0] = s;
}
int main() {
  unsigned long long* o;
  unsigned* sink;
  CK(cudaMalloc(&o, 8));
  CK(cudaMalloc(&sink, 4));
  unsigned long long h;
  auto one = [&](auto kern, int warps, int na, const char* name, const char* pt) {
    Stat s = repeat([&] {
      kern<<<1, 32 * warps>>>(0u, o, sink);
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(&h, o, 8, cudaMemcpyDeviceToHost));
      return (double)h / (ITERS * (double)na);
    });
    char p[48];
    snprintf(p, sizeof p, "%s_%s", name, pt);
    emit("mma_types", p, s, "cycles/instr/subpartition");
  };
#define POINT(T, name)                  \
  one(k<T, 16>, 1, 16, name, "w1");     \
  one(k<T, 16>, 4, 16, name, "w4");     \
  one(k<T, 1>, 1, 1, name, "chain");
  POINT(0, "f32_f16");
  POINT(1, "f16_f16");
  POINT(2, "f32_bf16");
  POINT(4, "f32_e4m3");
  POINT(5, "f16_e4m3");
  POINT(6, "f32_e5m2");
  POINT(7, "s32_s8");
  POINT(8, "s32_s4");
}

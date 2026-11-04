// Two warps on a sub-partition against one (question 29; HGEMM t256: replay -9 %, measured +1 %, the
// same tile at 128 threads): one tile's ldmatrix and mma per pass, split over T threads, a block
// barrier per pass, as TileLang's k-loop. T 128: L ldmatrix and M mma per warp, T 256: half of each
// (8 A sets either way). One block on one SM. Reported: cycles per pass (clock64 around the loop).
#include "../common.cuh"
typedef unsigned long long u64;
constexpr int PASSES = 32;
#define OPS4 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
__device__ __forceinline__ void mma(float* f, const unsigned* a, const unsigned* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " OPS4
               : "+f"(f[0]), "+f"(f[1]), "+f"(f[2]), "+f"(f[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void ldsm(unsigned* a, unsigned addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
               : "r"(addr));
}
template <int T, int L, int M>
__global__ void __launch_bounds__(T) k(u64* out, float* sink) {
  extern __shared__ __align__(128) unsigned char buf[];
  const unsigned sm = (unsigned)__cvta_generic_to_shared(buf);
  float f[8][4];
  unsigned a[8][4], b[2];
#pragma unroll
  for (int s = 0; s < 8; s++)
#pragma unroll
    for (int i = 0; i < 4; i++) a[s][i] = threadIdx.x * (i + s + 1);
#pragma unroll
  for (int i = 0; i < 2; i++) b[i] = threadIdx.x * (i + 5);
#pragma unroll
  for (int c = 0; c < 8; c++)
#pragma unroll
    for (int i = 0; i < 4; i++) f[c][i] = 0.f;
  __syncthreads();
  u64 c0 = clock64();
#pragma unroll 1
  for (int p = 0; p < PASSES; p++) {
#pragma unroll
    for (int c = 0; c < M; c++) {  // each ldmatrix feeds the mma after it
      if ((c + 1) * L / M > c * L / M) ldsm(a[c % 8], sm + (threadIdx.x & 31) * 16 + (c % 8) * 512);
      mma(f[c % 8], a[c % 8], b);
    }
    __syncthreads();
  }
  u64 c1 = clock64();
  if (threadIdx.x == 0) out[0] = c1 - c0;
  float s = 0;
#pragma unroll
  for (int c = 0; c < 8; c++) s += f[c][0] + f[c][1];
  if (s == 1.2345f) sink[0] = 1;
}
int main() {
  u64* o;
  float* sink;
  CK(cudaMalloc(&o, 8));
  CK(cudaMalloc(&sink, 4));
  auto point = [&](auto kern, int t, const char* name) {
    u64 h;
    Stat s = repeat([&] {
      kern<<<1, t, 4096>>>(o, sink);
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(&h, o, 8, cudaMemcpyDeviceToHost));
      return (double)h / PASSES;
    });
    emit("mma_ldsm", name, s, "cycles/pass");
  };
#define PT(T, L, M) point(k<T, L, M>, T, "t" #T "_L" #L "_M" #M)
  PT(64, 0, 128); PT(128, 0, 64); PT(256, 0, 32); PT(512, 0, 16);
  PT(64, 16, 128); PT(128, 8, 64); PT(256, 4, 32); PT(512, 2, 16);
  PT(64, 64, 128); PT(128, 32, 64); PT(256, 16, 32); PT(512, 8, 16);
}

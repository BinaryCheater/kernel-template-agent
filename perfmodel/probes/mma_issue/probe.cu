// What an mma stream pays on the issue side (round 6; STATE.md open question 3 and the systematic
// fast bias, questions 4 and 21; inventory §1 "Issue interference", §2 "mma issue beside FFMA"):
//   nN        N independent mma per loop pass (N = 2 .. 64), one warp: is the extra of mma_types (14
//             cycles on a pass of 16) per pass or per mma?
//   mK_M      8 mma per pass, M other instructions of kind K after each from the same warp
//             (K: fma, alu, lds, ldsm; M = 1 .. 14): does an instruction inside the hold cost?
//   nbrK      warp 0 runs n16 while warp 4 (same sub-partition) streams kind K, warps 1-3 idle.
// Reported: cycles per mma (clock64 of warp 0 around its loop).
#include "../common.cuh"
#define OPS4 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
__device__ __forceinline__ void mma(float* f, const unsigned* a, const unsigned* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " OPS4
               : "+f"(f[0]), "+f"(f[1]), "+f"(f[2]), "+f"(f[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
extern __shared__ __align__(128) unsigned char sbuf[];
struct Regs {
  float x[4];
  unsigned u[4];
  unsigned s;
};
template <int K>
__device__ __forceinline__ void other(Regs& r, int i) {
  if constexpr (K == 0) asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(r.x[i & 3]) : "f"(r.x[(i + 1) & 3]), "f"(r.x[(i + 2) & 3]));
  if constexpr (K == 1) asm volatile("xor.b32 %0, %0, %1;" : "+r"(r.u[i & 3]) : "r"(r.u[(i + 1) & 3]));
  if constexpr (K == 2) asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r.u[i & 3]) : "r"(r.s + 512 * (i & 3)) : "memory");
  if constexpr (K == 3) {
    unsigned q[4];
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(q[0]), "=r"(q[1]), "=r"(q[2]), "=r"(q[3]) : "r"(r.s + 512 * (i & 3)) : "memory");
    r.u[i & 3] ^= q[0];
  }
}
__device__ __forceinline__ void init(float (*f)[4], int na, unsigned* a, unsigned* b, unsigned z) {
  for (int i = 0; i < 4; i++) a[i] = threadIdx.x * (i + 1) + z;
  for (int i = 0; i < 2; i++) b[i] = threadIdx.x * (i + 5) + z;
}
template <int NA, int ITERS>
__global__ void pass(unsigned z, unsigned long long* out, unsigned* sink) {
  float f[16][4];
  unsigned a[4], b[2];
  init(f, 16, a, b, z);
#pragma unroll
  for (int c = 0; c < 16; c++)
#pragma unroll
    for (int i = 0; i < 4; i++) f[c][i] = 0.f;
  __syncthreads();
  long long t0 = clock64();
#pragma unroll 1
  for (int it = 0; it < ITERS; it++) {
#pragma unroll
    for (int c = 0; c < NA; c++) mma(f[c % 16], a, b);
  }
  __syncthreads();
  long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  float s = 0;
#pragma unroll
  for (int c = 0; c < 16; c++) s += f[c][0] + f[c][1];
  if (s == 1.2345f) sink[0] = 1;
}
template <int K, int M>
__global__ void beside(unsigned z, unsigned long long* out, unsigned* sink) {
  float f[8][4];
  unsigned a[4], b[2];
  init(f, 8, a, b, z);
#pragma unroll
  for (int c = 0; c < 8; c++)
#pragma unroll
    for (int i = 0; i < 4; i++) f[c][i] = 0.f;
  Regs r;
  r.s = (unsigned)__cvta_generic_to_shared(sbuf) + threadIdx.x * 16;
#pragma unroll
  for (int i = 0; i < 4; i++) r.x[i] = threadIdx.x + i, r.u[i] = threadIdx.x + i;
  __syncthreads();
  long long t0 = clock64();
#pragma unroll 1
  for (int it = 0; it < 256; it++) {
#pragma unroll
    for (int c = 0; c < 8; c++) {
      mma(f[c], a, b);
#pragma unroll
      for (int m = 0; m < M; m++) other<K>(r, m);
    }
  }
  __syncthreads();
  long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  float s = r.x[0] + r.x[1] + r.x[2] + r.x[3];
  unsigned v = r.u[0] ^ r.u[1] ^ r.u[2] ^ r.u[3];
#pragma unroll
  for (int c = 0; c < 8; c++) s += f[c][0];
  if (s == 1.2345f || v == 0x1234567u) sink[0] = 1;
}
// warp 0 runs 16 mma per pass, warp 4 (the same sub-partition) streams K, the others idle
template <int K>
__global__ void nbr(unsigned z, unsigned long long* out, unsigned* sink) {
  int w = threadIdx.x / 32;
  float f[16][4];
  unsigned a[4], b[2];
  init(f, 16, a, b, z);
#pragma unroll
  for (int c = 0; c < 16; c++)
#pragma unroll
    for (int i = 0; i < 4; i++) f[c][i] = 0.f;
  Regs r;
  r.s = (unsigned)__cvta_generic_to_shared(sbuf) + threadIdx.x * 16;
#pragma unroll
  for (int i = 0; i < 4; i++) r.x[i] = threadIdx.x + i, r.u[i] = threadIdx.x + i;
  __syncthreads();
  if (w == 0) {
    long long t0 = clock64();
#pragma unroll 1
    for (int it = 0; it < 256; it++) {
#pragma unroll
      for (int c = 0; c < 16; c++) mma(f[c], a, b);
    }
    long long t1 = clock64();
    if (threadIdx.x == 0) out[0] = t1 - t0;
  } else if (w == 4 && K >= 0) {
#pragma unroll 1
    for (int it = 0; it < 8192; it++) {
#pragma unroll
      for (int m = 0; m < 8; m++) other<(K < 0 ? 1 : K)>(r, m);
    }
  }
  float s = r.x[0] + r.x[1] + r.x[2] + r.x[3];
  unsigned v = r.u[0] ^ r.u[1] ^ r.u[2] ^ r.u[3];
#pragma unroll
  for (int c = 0; c < 16; c++) s += f[c][0];
  if (s == 1.2345f || v == 0x1234567u) sink[0] = 1;
}
static unsigned long long* o;
static unsigned* sink;
template <class F>
static void point(const char* p, double per, F launch) {
  unsigned long long h;
  Stat s = repeat([&] {
    launch();
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(&h, o, 8, cudaMemcpyDeviceToHost));
    return (double)h / per;
  });
  emit("mma_issue", p, s, "cycles/mma");
}
template <int NA, int ITERS>
static void n_point(const char* p) {
  point(p, (double)NA * ITERS, [&] { pass<NA, ITERS><<<1, 32>>>(0u, o, sink); });
}
template <int K, int M>
static void b_point(const char* kind) {
  char p[32];
  snprintf(p, sizeof p, "m%s_%d", kind, M);
  CK(cudaFuncSetAttribute(beside<K, M>, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384));
  point(p, 8.0 * 256, [&] { beside<K, M><<<1, 32, 16384>>>(0u, o, sink); });
}
template <int K>
static void b_sweep(const char* kind) {
  b_point<K, 1>(kind);
  b_point<K, 2>(kind);
  b_point<K, 4>(kind);
  b_point<K, 8>(kind);
  b_point<K, 14>(kind);
}
template <int K>
static void nb_point(const char* p) {
  CK(cudaFuncSetAttribute(nbr<K>, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384));
  point(p, 16.0 * 256, [&] { nbr<K><<<1, 160, 16384>>>(0u, o, sink); });
}
int main() {
  CK(cudaMalloc(&o, 8));
  CK(cudaMalloc(&sink, 4));
  n_point<2, 2048>("n2");
  n_point<4, 1024>("n4");
  n_point<8, 512>("n8");
  n_point<16, 256>("n16");
  n_point<32, 128>("n32");
  n_point<64, 64>("n64");
  n_point<3, 1024>("n3");
  n_point<5, 1024>("n5");
  b_sweep<0>("fma");
  b_sweep<1>("alu");
  b_sweep<2>("lds");
  b_point<3, 1>("ldsm");
  b_point<3, 2>("ldsm");
  nb_point<-1>("nbr_none");
  nb_point<0>("nbr_fma");
  nb_point<1>("nbr_alu");
  nb_point<2>("nbr_lds");
}

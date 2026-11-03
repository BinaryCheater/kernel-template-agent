// The tensor pipe's rate by stream (round 6, question 22: mma_issue's one-warp stream runs 17 cycles
// per mma, mma_pair's 16, the HGEMM rows replay right only at 17). Two things differ: the compiler
// (the host's nvcc spaces mma 16 cycles apart, TileLang's image nvcc 12, so the stream queues at the
// pipe) and the operands (the same registers, or fresh ones per mma as after ldmatrix). This source
// is built by both compilers. 16 independent mma per pass, operands cycling through SA sets of A and
// SB sets of B; 1 warp and 8 warps (2 per sub-partition, the pipe-limited stream). Reported: cycles
// per mma per sub-partition (clock64 around the loop between two block barriers).
#include "../common.cuh"
constexpr int ITERS = 256;
#define OPS4 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
__device__ __forceinline__ void mma(float* f, const unsigned* a, const unsigned* b) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 " OPS4
               : "+f"(f[0]), "+f"(f[1]), "+f"(f[2]), "+f"(f[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
template <int SA, int SB>
__global__ void k(unsigned z, unsigned long long* out, unsigned* sink) {
  float f[16][4];
  unsigned a[SA][4], b[SB][2];
#pragma unroll
  for (int s = 0; s < SA; s++)
#pragma unroll
    for (int i = 0; i < 4; i++) a[s][i] = threadIdx.x * (i + s + 1) + z;
#pragma unroll
  for (int s = 0; s < SB; s++)
#pragma unroll
    for (int i = 0; i < 2; i++) b[s][i] = threadIdx.x * (i + s + 5) + z;
#pragma unroll
  for (int c = 0; c < 16; c++)
#pragma unroll
    for (int i = 0; i < 4; i++) f[c][i] = 0.f;
  __syncthreads();
  long long t0 = clock64();
#pragma unroll 1
  for (int it = 0; it < ITERS; it++) {
#pragma unroll
    for (int c = 0; c < 16; c++) mma(f[c], a[c % SA], b[c % SB]);
  }
  __syncthreads();
  long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  float s = 0;
#pragma unroll
  for (int c = 0; c < 16; c++) s += f[c][0] + f[c][1];
  if (s == 1.2345f) sink[0] = 1;
}
static unsigned long long* o;
static unsigned* sink;
template <int SA, int SB>
static void point() {
  for (int w : {1, 8}) {
    unsigned long long h;
    Stat s = repeat([&] {
      k<SA, SB><<<1, 32 * w>>>(0u, o, sink);
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(&h, o, 8, cudaMemcpyDeviceToHost));
      return (double)h / (ITERS * 16.0 * (w / 4 ? w / 4 : 1));
    });
    char p[48];
    snprintf(p, sizeof p, "v13_%d_a%d_b%d_w%d", (int)__CUDACC_VER_MINOR__, SA, SB, w);
    emit("mma_pipe", p, s, "cycles/mma/subpartition");
  }
}
int main() {
  CK(cudaMalloc(&o, 8));
  CK(cudaMalloc(&sink, 4));
  point<1, 1>();
  point<2, 1>();
  point<4, 1>();
  point<8, 1>();
  point<1, 2>();
  point<1, 4>();
  point<1, 8>();
  point<2, 2>();
  point<4, 4>();
  point<8, 8>();
}

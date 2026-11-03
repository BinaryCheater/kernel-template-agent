// The L1 hit path of plain global loads (inventory §5 "L1 hit path: 2 cycles per line, 45-cycle
// latency": measured upstream by ldg_atomic.cu, hit/miss supplied by the driver; re-measured here
// for the decode/GEMV/norm kernels). Pattern copied from upstream ldg_atomic.cu.
//  chase:  one lane, dependent ld.global.ca through a 4 KB ring (L1-resident): cycles per hop.
//  tput_wW: W warps on one SM, ld.global.ca.v4 (512 contiguous bytes per warp instruction) from a
//          16 KB set (L1 hits after the first pass), 8 independent loads per pass: cycles per warp
//          instruction per SM.
#include "../common.cuh"
constexpr int HOPS = 2048, ITERS = 256, RING = 1024, SET4 = 1024;  // RING words; SET4 int4s = 16 KB
__global__ void chase(const unsigned* next, unsigned long long* out) {
  unsigned p = 0;
#pragma unroll 1
  for (int i = 0; i < 64; i++) asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(p) : "l"(next + p));
  long long t0 = clock64();
#pragma unroll 1
  for (int i = 0; i < HOPS; i++) asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(p) : "l"(next + p));
  long long t1 = clock64();
  out[0] = t1 - t0;
  out[1] = p;
}
__global__ void tput(const int4* src, unsigned long long* out, int* sink) {
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  int4 acc = make_int4(0, 0, 0, 0);
  for (int w = warp; w < SET4 / 32; w += blockDim.x / 32) {  // warm L1
    int4 v;
    asm volatile("ld.global.ca.v4.s32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(src + w * 32 + lane));
    acc.x ^= v.x;
  }
  __syncthreads();
  long long t0 = clock64();
#pragma unroll 1
  for (int i = 0; i < ITERS; i++) {
#pragma unroll
    for (int u = 0; u < 8; u++) {
      int w = (warp * 3 + i * 8 + u) & (SET4 / 32 - 1);
      int4 v;
      asm volatile("ld.global.ca.v4.s32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(src + w * 32 + lane + (acc.x & 0)));
      acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
  }
  __syncthreads();
  long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  if (acc.x == 12345 && acc.y == acc.z) sink[0] = acc.w;
}
int main() {
  unsigned *ring, hr[RING];
  int4* set;
  int* sink;
  unsigned long long* o;
  unsigned long long h;
  for (int i = 0; i < RING; i++) hr[i] = (i + 33) % RING;  // stride 132 B: a new line every hop
  CK(cudaMalloc(&ring, sizeof hr));
  CK(cudaMemcpy(ring, hr, sizeof hr, cudaMemcpyHostToDevice));
  CK(cudaMalloc(&set, SET4 * 16));
  CK(cudaMemset(set, 0, SET4 * 16));
  CK(cudaMalloc(&sink, 4));
  CK(cudaMalloc(&o, 16));
  Stat s = repeat([&] {
    chase<<<1, 1>>>(ring, o);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(&h, o, 8, cudaMemcpyDeviceToHost));
    return (double)h / HOPS;
  });
  emit("l1hit", "chase", s, "cycles/hop");
  for (int w : {4, 8, 16, 32}) {
    Stat t = repeat([&] {
      tput<<<1, 32 * w>>>(set, o, sink);
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(&h, o, 8, cudaMemcpyDeviceToHost));
      return (double)h / (ITERS * 8.0 * w);
    });
    char p[32];
    snprintf(p, sizeof p, "tput_w%d", w);
    emit("l1hit", p, t, "cycles/instr/SM");
  }
}

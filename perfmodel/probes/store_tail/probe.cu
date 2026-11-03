// P-1: the C-store path at partial and full grids (STATE.md P-1). Copied from upstream
// agentic/tilelang/microbench/storepat.cu (research repo, read-only): a 256 x 128 fp16 C tile
// (64 KB per block) stored into a 1024-column matrix, 256 threads, blocks take tiles columns first.
//  tl:  TileLang's epilogue pattern: 4 B per lane, 8 rows x 16 B per instruction (64 STG per thread)
//  v16: 16 B per lane, 4 rows x 128 B per instruction (16 STG per thread)
// Reported: per-block %globaltimer span (first instruction to after the closing barrier), mean over
// the blocks of one launch, median over REPS launches; blocks in {1, 2, 8, 16, 32, 48}, one per SM.
#include "../common.cuh"
__device__ unsigned long long sp[1024];
template <int MODE>
__global__ void st(unsigned* C, int tiles_n) {
  const int by = blockIdx.x / tiles_n, bx = blockIdx.x % tiles_n, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
  unsigned base = by * 256 * 512 + bx * 64;
  unsigned long long g0, g1;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  if (MODE == 0) {
#pragma unroll 4
    for (int i = 0; i < 64; i++) {
      unsigned r = (w >> 1) * 64 + (i >> 4) * 16 + (i & 1) * 8 + (lane >> 2), c = (w & 1) * 32 + ((i >> 1) & 7) * 4 + (lane & 3);
      C[base + r * 512 + c] = i;
    }
  } else {
#pragma unroll 4
    for (int i = 0; i < 16; i++) {
      unsigned r = w * 32 + i * 2 + (lane >> 4), c = (lane & 15) * 4;
      uint4 v = make_uint4(i, i, i, i);
      *(uint4*)(C + base + r * 512 + c) = v;
    }
  }
  __syncthreads();
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
  if (tid == 0) sp[blockIdx.x] = g1 - g0;
}
template <int MODE>
void run(unsigned* C, int blocks, const char* name) {
  static unsigned long long h[1024];
  Stat s = repeat([&] {
    st<MODE><<<blocks, 256>>>(C, 8);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpyFromSymbol(h, sp, 8 * blocks));
    double m = 0;
    for (int b = 0; b < blocks; b++) m += h[b];
    return m / blocks / 1e3;
  });
  char p[32];
  snprintf(p, sizeof p, "%s_b%d", name, blocks);
  emit("store_tail", p, s, "us");
}
int main() {
  unsigned* C;
  CK(cudaMalloc(&C, 64 << 20));
  for (int b : {1, 2, 8, 16, 32, 48}) {
    run<0>(C, b, "tl");
    run<1>(C, b, "v16");
  }
}

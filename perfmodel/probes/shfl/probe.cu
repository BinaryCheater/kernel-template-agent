// Warp shuffles (inventory §3 "Shuffles SHFL": absent). CH independent chains per thread of
// shfl.sync.bfly with lane masks cycling 1, 2, 4, 8, 16 (the reduction pattern of softmax and RMSNorm
// row reductions); each shuffle's source is the previous result of its chain. CH = 1 on one warp:
// latency per hop; CH = 8 on 1-32 warps of one SM: cycles per warp instruction per SM.
#include "../common.cuh"
constexpr int ITERS = 256;
template <int CH>
__global__ void k(unsigned long long* out, unsigned* sink) {
  unsigned v[CH];
#pragma unroll
  for (int c = 0; c < CH; c++) v[c] = threadIdx.x * (c + 1);
  __syncthreads();
  long long t0 = clock64();
#pragma unroll 1
  for (int i = 0; i < ITERS; i++) {
#pragma unroll
    for (int u = 0; u < 5; u++)
#pragma unroll
      for (int c = 0; c < CH; c++) v[c] = __shfl_xor_sync(0xffffffffu, v[c], 1 << u);
  }
  __syncthreads();  // every warp done: the block's time, not warp 0's
  long long t1 = clock64();
  if (threadIdx.x == 0) out[0] = t1 - t0;
  unsigned s = 0;
#pragma unroll
  for (int c = 0; c < CH; c++) s ^= v[c];
  if (s == 0x12345u) sink[0] = s;
}
int main() {
  unsigned long long* o;
  unsigned* sink;
  CK(cudaMalloc(&o, 8));
  CK(cudaMalloc(&sink, 4));
  unsigned long long h;
  auto one = [&](auto kern, int warps, int ch, const char* p) {
    Stat s = repeat([&] {
      kern<<<1, 32 * warps>>>(o, sink);
      CK(cudaDeviceSynchronize());
      CK(cudaMemcpy(&h, o, 8, cudaMemcpyDeviceToHost));
      return (double)h / (ITERS * 5.0 * ch * (ch == 1 ? 1 : warps));
    });
    emit("shfl", p, s, ch == 1 ? "cycles/hop" : "cycles/instr/SM");
  };
  one(k<1>, 1, 1, "latency_w1");
  for (int w : {1, 4, 8, 16, 32}) {
    char p[32];
    snprintf(p, sizeof p, "tput_w%d", w);
    one(k<8>, w, 8, p);
  }
}

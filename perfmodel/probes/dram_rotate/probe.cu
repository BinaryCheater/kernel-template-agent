// DRAM reads under rotating inputs (inventory §5 "DRAM read rate by run length", "DRAM queue per SM",
// "DRAM latency spread": the decode-style streaming case, cache state `rotating`). Four 63.75 MB buffers
// are read in turn (one per launch, 256 MB in rotation > the 24 MB L2), each element once, LDG.128,
// one load in flight per warp (#pragma unroll 1: the xor of a load is consumed before the next
// load issues), threads per block T in {32, 64, 128, 256, 512, 1024}, 48 blocks: bytes in flight
// on the chip grow 48 x T x 16 B. Reported: event time per launch and GB/s.
#include "../common.cuh"
constexpr size_t BYTES = 48ull * 1024 * 16 * 85, N4 = BYTES / 16;  // 63.75 MB, divisible by 48 x T x 16
template <int T>
__global__ void __launch_bounds__(T) stream(const int4* __restrict__ src, int* sink) {
  constexpr int STEPS = N4 / (48 * T);  // compile-time trip count (48 blocks of T threads)
  int acc = 0;
  const int4* p = src + blockIdx.x * T + threadIdx.x;
#pragma unroll 1
  for (int i = 0; i < STEPS; i++) {
    int4 v = __ldcs(p + (size_t)i * 48 * T);
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x7fff1234) sink[0] = acc;
}
int main() {
  int4* buf[4];
  int* sink;
  for (auto& b : buf) {
    CK(cudaMalloc(&b, BYTES));
    CK(cudaMemset(b, 1, BYTES));
  }
  CK(cudaMalloc(&sink, 4));
  auto one = [&](auto kern, int t) {
    int r = 0;
    Stat s = repeat([&] { return event_us([&] { kern<<<48, t>>>(buf[r++ % 4], sink); }); });
    char p[32], x[64];
    snprintf(p, sizeof p, "t%d", t);
    snprintf(x, sizeof x, "GBs=%.1f inflight_KB=%.0f", BYTES / (s.median * 1e3), 48.0 * t * 16 / 1024);
    emit("dram_rotate", p, s, "us", x);
  };
  one(stream<32>, 32);
  one(stream<64>, 64);
  one(stream<128>, 128);
  one(stream<256>, 256);
  one(stream<512>, 512);
  one(stream<1024>, 1024);
}

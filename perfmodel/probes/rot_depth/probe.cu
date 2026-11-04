// What a rotating launch leaves in the L2 (open questions 12, 24; r7's rotation depth rows: decode
// 68.9 us at depth 67 MB, 78 us at 134 and 268 MB, replay 78.8; RMSNorm without stores 20.7 us at 33
// MB): the dram_burst pattern, n x 786 KB per launch, the start walks a window of D MiB (one read of
// each chunk per turn), after three turns untimed. D = n x 786 KB: every launch reads what it read
// last turn; D above the L2: rotating. Reported: event time per launch.
#include "../common.cuh"
constexpr size_t UNIT = 48ull * 1024 * 16;
constexpr int T = 1024;
template <int N>
__global__ void __launch_bounds__(T) stream(const int4* __restrict__ src, int* sink) {
  constexpr int STEPS = UNIT * N / 16 / (48 * T);
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
  const size_t TOTAL = 512ull << 20;
  int4* buf;
  int* sink;
  CK(cudaMalloc(&buf, TOTAL));
  CK(cudaMemset(buf, 1, TOTAL));
  CK(cudaMalloc(&sink, 4));
  auto one = [&](auto kern, int n, int mib) {
    size_t at = 0, chunk = UNIT * n, window = (size_t)mib << 20;
    if (window < chunk) window = chunk;
    auto launch = [&] {
      if (at + chunk > window) at = 0;
      const int4* src = buf + at / 16;
      at += chunk;
      return event_us([&] { kern<<<48, T>>>(src, sink); });
    };
    for (size_t i = 0; i < 3 * window / chunk + 2; i++) launch();
    Stat s = repeat(launch);
    char p[32];
    snprintf(p, sizeof p, "n%d_D%d", n, mib);
    emit("rot_depth", p, s, "us");
  };
  for (int mib : {1, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384}) {
    one(stream<10>, 10, mib);
    one(stream<21>, 21, mib);
  }
}

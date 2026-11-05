// round 20: the scale table's load every 1 passes, i5, without the stream
extern "C" __global__ void __launch_bounds__(128) k_mem(unsigned* o, const char* a, const unsigned short* c) {
  const unsigned row = blockIdx.x * 4 + threadIdx.y, kr = threadIdx.x;
  unsigned s = 0;
  for (unsigned ko = 0; ko < 11; ++ko) {
    const size_t at = (size_t)row * 1344 + (ko * 32 + kr) * 4;
    
    if (ko % 1 == 0) { 
      if (kr % 5 == 0) s ^= c[row * 48 + (ko * 256 + kr * 8) / 56];
      if (kr % 5 == 1) s ^= c[row * 48 + (ko * 256 + kr * 8) / 56];
      if (kr % 5 == 2) s ^= c[row * 48 + (ko * 256 + kr * 8) / 56];
      if (kr % 5 == 3) s ^= c[row * 48 + (ko * 256 + kr * 8) / 56];
      if (kr % 5 == 4) s ^= c[row * 48 + (ko * 256 + kr * 8) / 56]; }
  }
  if (kr == 0) o[row & 65535u] = s;
}

// All-pairs forces: each thread one body against 512 others, reciprocal square roots.
__global__ void __launch_bounds__(256) k_nbody(float4* acc, const float4* pos) {
  const int i = blockIdx.x * 256 + threadIdx.x;
  const float4 p = pos[i];
  float ax = 0.f, ay = 0.f, az = 0.f;
  for (int k = 0; k < 512; ++k) {
    const float4 q = pos[(i + k * 7) & 16383];
    const float dx = q.x - p.x, dy = q.y - p.y, dz = q.z - p.z;
    const float r2 = dx * dx + dy * dy + dz * dz + 0.01f;
    const float s = rsqrtf(r2);
    const float w = q.w * s * s * s;
    ax += dx * w; ay += dy * w; az += dz * w;
  }
  acc[i] = make_float4(ax, ay, az, 0.f);
}

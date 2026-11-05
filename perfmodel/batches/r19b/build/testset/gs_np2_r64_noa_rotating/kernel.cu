#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/reduce.h>
#include <tl_templates/cuda/scan.h>
#include <tl_templates/cuda/ldsm.h>
#include <tl_templates/cuda/threadblock_swizzle.h>
#include <tl_templates/cuda/debug.h>
#ifdef ENABLE_BF16
#include <tl_templates/cuda/cuda_bf16_fallbacks.cuh>
#endif

extern "C" __global__ void gemv_kernel(const signed char* __restrict__ Bq, half_t* __restrict__ C, const half_t* __restrict__ Sc);
extern "C" __global__ void __launch_bounds__(128, 1) gemv_kernel(const signed char* __restrict__ Bq, half_t* __restrict__ C, const half_t* __restrict__ Sc) {
  float acc[1];
  half_t a[8];
  signed char bq[4];
  half_t bd[8];
  float red[1];
  acc[0] = 0x0p+0f/*0.000000e+00*/;
  for (int ko = 0; ko < 5; ++ko) {
    for (int v = 0; v < 2; ++v) {
      uint2 __1;
      int4 v_ = make_int4((((v * 4) + ((int)threadIdx.x)))+(1*0), (((v * 4) + ((int)threadIdx.x)))+(1*1), (((v * 4) + ((int)threadIdx.x)))+(1*2), (((v * 4) + ((int)threadIdx.x)))+(1*3));
      ((half2*)(&(__1.x)))->x = (half_t)(v_.x);
      ((half2*)(&(__1.x)))->y = (half_t)(v_.y);
      ((half2*)(&(__1.y)))->x = (half_t)(v_.z);
      ((half2*)(&(__1.y)))->y = (half_t)(v_.w);
      *(uint2*)(a + (v * 4)) = __1;
    }
    *(int*)(bq + 0) = *(int*)(Bq + ((((((int)blockIdx.x) * 2560) + (((int)threadIdx.y) * 1280)) + (ko * 256)) + (((int)threadIdx.x) * 4)));
    for (int ki = 0; ki < 8; ++ki) {
      bd[ki] = ((((half_t)((((int)bq[(ki >> 1)]) >> ((ki & 1) * 4)) & 15)) - half_t(0x1p+3f/*8.000000e+00*/)) * Sc[((((((int)blockIdx.x) * 160) + (((int)threadIdx.y) * 80)) + (ko * 16)) + (((int)threadIdx.x) >> 2))]);
    }
    for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
      acc[0] = (acc[0] + (((float)a[ki_1]) * ((float)bd[ki_1])));
    }
  }
  __shared__ __align__(16) float red_buf_staging[4];
  float red_buf0[1];
  float t0[1];
  uint mask[1];
  float red_buf0_1[1];
  float t0_1[1];
  uint mask_1[1];
  __shared__ __align__(16) float red_result[2];
  red_buf0[0] = acc[0];
  mask[0] = __activemask();
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 16, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  if ((((int)threadIdx.x) % 32) == 0) {
    red_buf_staging[((((int)threadIdx.y) * 2) + (((int)threadIdx.x) >> 5))] = red_buf0[0];
  }
  __syncthreads();
  if (((int)threadIdx.x) < 2) {
    red_buf0_1[0] = red_buf_staging[((((int)threadIdx.y) * 2) + ((int)threadIdx.x))];
  }
  mask_1[0] = __activemask();
  t0_1[0] = __shfl_down_sync(mask_1[0], red_buf0_1[0], 1, 32);
  red_buf0_1[0] = (red_buf0_1[0] + t0_1[0]);
  if (((int)threadIdx.x) == 0) {
    red_result[((int)threadIdx.y)] = red_buf0_1[0];
  }
  __syncthreads();
  if (((int)threadIdx.x) == 0) {
    C[((((int)blockIdx.x) * 2) + ((int)threadIdx.y))] = ((half_t)red_result[((int)threadIdx.y)]);
  }
}


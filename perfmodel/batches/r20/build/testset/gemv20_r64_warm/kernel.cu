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

extern "C" __global__ void gemv_kernel(const half_t* __restrict__ A, const signed char* __restrict__ Bq, half_t* __restrict__ C, const half_t* __restrict__ Sc);
extern "C" __global__ void __launch_bounds__(256, 1) gemv_kernel(const half_t* __restrict__ A, const signed char* __restrict__ Bq, half_t* __restrict__ C, const half_t* __restrict__ Sc) {
  float acc[1];
  half_t a[8];
  signed char bq[4];
  half_t bd[8];
  float red[1];
  acc[0] = 0x0p+0f/*0.000000e+00*/;
  for (int ko = 0; ko < 6; ++ko) {
    half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
    uint4 condval;
    if ((((ko * 8) + (((int)threadIdx.x) >> 3)) < 45)) {
      condval = *(uint4*)(A + ((ko * 512) + (((int)threadIdx.x) * 8)));
    } else {
      condval = make_uint4(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
    }
    *(uint4*)(a + 0) = condval;
    signed char broadcast_var_1 = (signed char)0;
    int condval_1;
    if ((((ko * 8) + (((int)threadIdx.x) >> 3)) < 45)) {
      condval_1 = *(int*)(Bq + ((((((int)blockIdx.x) * 5760) + (((int)threadIdx.y) * 1440)) + (ko * 256)) + (((int)threadIdx.x) * 4)));
    } else {
      condval_1 = tl::broadcast<int>(broadcast_var_1);
    }
    *(int*)(bq + 0) = condval_1;
    for (int ki = 0; ki < 8; ++ki) {
      half_t condval_2;
      if ((((ko * 8) + (((int)threadIdx.x) >> 3)) < 45)) {
        condval_2 = Sc[(((((int)blockIdx.x) * 144) + (((int)threadIdx.y) * 36)) + (((ko * 32) + (((int)threadIdx.x) >> 1)) / 5))];
      } else {
        condval_2 = half_t(0x0p+0f/*0.000000e+00*/);
      }
      bd[ki] = ((((half_t)((((int)bq[(ki >> 1)]) >> ((ki & 1) * 4)) & 15)) - half_t(0x1p+3f/*8.000000e+00*/)) * condval_2);
    }
    for (int ki_1 = 0; ki_1 < 8; ++ki_1) {
      acc[0] = (acc[0] + (((float)a[ki_1]) * ((float)bd[ki_1])));
    }
  }
  __shared__ __align__(16) float red_buf_staging[8];
  float red_buf0[1];
  float t0[1];
  uint mask[1];
  float red_buf0_1[1];
  float t0_1[1];
  uint mask_1[1];
  __shared__ __align__(16) float red_result[4];
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
    C[((((int)blockIdx.x) * 4) + ((int)threadIdx.y))] = ((half_t)red_result[((int)threadIdx.y)]);
  }
}


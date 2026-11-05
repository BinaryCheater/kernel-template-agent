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
extern "C" __global__ void __launch_bounds__(128, 1) gemv_kernel(const half_t* __restrict__ A, const signed char* __restrict__ Bq, half_t* __restrict__ C, const half_t* __restrict__ Sc) {
  float acc[1];
  half_t a[4];
  signed char bq[2];
  half_t bd[4];
  float red[1];
  acc[0] = 0x0p+0f/*0.000000e+00*/;
  for (int ko = 0; ko < 30; ++ko) {
    half_t broadcast_var = half_t(0x0p+0f/*0.000000e+00*/);
    uint2 condval;
    if ((((ko * 4) + (((int)threadIdx.x) >> 3)) < 117)) {
      condval = *(uint2*)(A + ((ko * 128) + (((int)threadIdx.x) * 4)));
    } else {
      condval = make_uint2(__pack_half2(broadcast_var, broadcast_var), __pack_half2(broadcast_var, broadcast_var));
    }
    *(uint2*)(a + 0) = condval;
    signed char broadcast_var_1 = (signed char)0;
    char2 condval_1;
    if ((((ko * 4) + (((int)threadIdx.x) >> 3)) < 117)) {
      condval_1 = *(char2*)(Bq + ((((((int)blockIdx.x) * 7488) + (((int)threadIdx.y) * 1872)) + (ko * 64)) + (((int)threadIdx.x) * 2)));
    } else {
      condval_1 = make_char2(broadcast_var_1, broadcast_var_1);
    }
    *(char2*)(bq + 0) = condval_1;
    for (int ki = 0; ki < 4; ++ki) {
      half_t condval_2;
      if ((((ko * 4) + (((int)threadIdx.x) >> 3)) < 117)) {
        condval_2 = Sc[(((((int)blockIdx.x) * 144) + (((int)threadIdx.y) * 36)) + (((ko * 16) + (((int)threadIdx.x) >> 1)) / 13))];
      } else {
        condval_2 = half_t(0x0p+0f/*0.000000e+00*/);
      }
      bd[ki] = ((((half_t)((((int)bq[(ki >> 1)]) >> ((ki & 1) * 4)) & 15)) - half_t(0x1p+3f/*8.000000e+00*/)) * condval_2);
    }
    for (int ki_1 = 0; ki_1 < 4; ++ki_1) {
      acc[0] = (acc[0] + (((float)a[ki_1]) * ((float)bd[ki_1])));
    }
  }
  float red_buf0[1];
  float t0[1];
  uint mask[1];
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
  red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], (((int)threadIdx.y) * 32), 32);
  if (((int)threadIdx.x) == 0) {
    C[((((int)blockIdx.x) * 4) + ((int)threadIdx.y))] = ((half_t)red_buf0[0]);
  }
}


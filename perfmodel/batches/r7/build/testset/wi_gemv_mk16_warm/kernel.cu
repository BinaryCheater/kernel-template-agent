#if defined(_MSC_VER) && !defined(__clang__) && _MSC_VER < 1940
#define _tl_orig_alignas alignas
#define alignas(N) _tl_orig_alignas((N) <= 64 ? (N) : 64)
#include <cuda.h>
#undef alignas
#define alignas _tl_orig_alignas
#endif
#include <tl_templates/cuda/copy_sm90.h>
#include <tl_templates/cuda/copy_sm100.h>
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
  half_t a[16];
  signed char bq[8];
  half_t bd[16];
  float red[1];
  acc[0] = 0x0p+0f/*0.000000e+00*/;
  for (int ko = 0; ko < 8; ++ko) {
    *(ulonglong4*)(a + 0) = tl::load_global_256(&(*(ulonglong4*)(A + ((ko * 512) + (((int)threadIdx.x) * 16)))));
    *(int2*)(bq + 0) = *(int2*)(Bq + ((((((int)blockIdx.x) * 8192) + (((int)threadIdx.y) * 2048)) + (ko * 256)) + (((int)threadIdx.x) * 8)));
    for (int ki = 0; ki < 16; ++ki) {
      bd[ki] = ((((half_t)((((int)bq[(ki >> 1)]) >> ((ki & 1) * 4)) & 15)) - half_t(0x1p+3f/*8.000000e+00*/)) * Sc[((((((int)blockIdx.x) * 128) + (((int)threadIdx.y) * 32)) + (ko * 4)) + (((int)threadIdx.x) >> 3))]);
    }
    for (int ki_1 = 0; ki_1 < 16; ++ki_1) {
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


#!/bin/sh
# TileLang build job, run inside the worker image (kta.execution.sandbox).
# 1. gen.py lowers the prim_func KTA_ENTRY of kernel.py to CUDA source (no GPU needed).
# 2. The host nvcc, mounted at {cuda_home}, compiles it to a cubin with TileLang's headers,
#    so TileLang and CUDA C++ modules come from the same toolchain (decision D187).
set -eu
python gen.py kernel.py "$KTA_ENTRY" "$KTA_ARCH" > out/gen.json
TL=$(python -c 'import os, tilelang; print(os.path.dirname(tilelang.__file__))')
# shellcheck disable=SC2086  # KTA_NVCC_FLAGS is a word list
"$KTA_CUDA_HOME/bin/nvcc" -cubin $KTA_NVCC_FLAGS -I"$TL/src" -I"$TL/3rdparty/cutlass/include" \
  -Xptxas -v -o out/module.cubin out/kernel.cu
"$KTA_CUDA_HOME/bin/cuobjdump" -sass out/module.cubin > out/module.sass

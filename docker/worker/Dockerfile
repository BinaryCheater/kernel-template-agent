# KTA worker image: external kernel toolchains that KTA runs only inside containers.
# Phase 1 holds TileLang (codegen only; compiling uses the host CUDA toolkit mounted at
# /opt/kta-cuda, see src/kta/execution/sandbox.py). Build with docker/build.
#
# Base: NVIDIA PyTorch 26.03 (Ubuntu 24.04, Python 3.12, aarch64), pinned by digest.
FROM nvcr.io/nvidia/pytorch@sha256:3c90e38f5ec51e51d1c73bd7eb3d83674a254f451147c5cadc4344314258a112

COPY requirements.lock /tmp/kta-requirements.lock
RUN pip install --no-cache-dir --no-deps -r /tmp/kta-requirements.lock \
    && python -c "import tilelang; assert tilelang.__version__.startswith('0.1.15'), tilelang.__version__" \
    && rm /tmp/kta-requirements.lock

# Read by the worker probe (kta.execution.workerprobe) to label backend.tilelang.
LABEL org.kta.tilelang="0.1.15"
# The NGC entrypoint prints a banner and checks the GPU; KTA jobs run plain commands.
ENTRYPOINT []

#!/usr/bin/env bash
# Build every probe, the protocol kernels and the test-set kernels on spark (perfmodel-measure build).
#   perfmodel/remote/build.sh <out_dir>
# Run from the root of a deployed tree. For each unit: the executable, the cubin, the SASS with
# control bits (`nvdisasm -hex` of the cubin, the form gb10sim.sass parses), the ptxas resource
# report. Probes and protocol kernels use the host CUDA (13.0); TileLang kernels are generated and
# compiled inside the worker image (its TileLang headers), with the image's nvcc.
#   ONLY=<regex> limits the probes and test-set kernels to the matching names (protocol always).
set -euo pipefail
out="$(mkdir -p "$1" && cd "$1" && pwd)"
only="${ONLY:-.}"
root="$(pwd)"
image="kta-worker:tilelang-0.1.15"
flags=(-O3 -arch=sm_121a -std=c++17)
nvcc --version | tail -2 > "$out/nvcc-host.txt"

one() { # <name> <src> [extra nvcc flags...] (host nvcc)
  local name="$1" src="$2"; shift 2
  local d="$out/$name"; mkdir -p "$d"
  nvcc "${flags[@]}" "$@" -Xptxas -v -o "$d/bin" "$src" 2> "$d/ptxas.txt"
  nvcc "${flags[@]}" "$@" -cubin -o "$d/k.cubin" "$src"
  nvdisasm -hex "$d/k.cubin" > "$d/k.sass"
  cuobjdump -res-usage "$d/k.cubin" > "$d/res-usage.txt"
}

one protocol perfmodel/protocol/protocol.cu
for p in perfmodel/probes/*/; do
  n="$(basename "$p")"
  [ "$n" = l2share_copy ] && continue   # TileLang headers: built in the image below
  echo "probe-$n" | grep -Eq "$only" || continue
  one "probe-$n" "$p/probe.cu"
done

# TileLang: generate the test set, then compile it and the TileLang-based probe in the image
docker run --rm --gpus all -u "$(id -u):$(id -g)" -e HOME=/tmp -e ONLY="$only" -v "$root":/w -v "$out":/o -w /w "$image" sh -ec '
  python perfmodel/testset/gen.py /o/testset > /o/testset-gen.log 2>&1 || { tail -40 /o/testset-gen.log; exit 1; }
  tl=$(python -c "import tilelang, os; print(os.path.dirname(tilelang.__file__))")
  inc="-I$tl/src -I$tl/3rdparty/cutlass/include"
  f="-O3 -arch=sm_121a -std=c++20 --expt-relaxed-constexpr -lcuda $inc"
  nvcc --version | tail -2 > /o/nvcc-image.txt
  python -c "import tilelang; print(tilelang.__version__)" > /o/tilelang-version.txt
  build() { d=$1; src=$2
    nvcc $f -Xptxas -v -o $d/bin $src 2> $d/ptxas.txt
    nvcc $f -cubin -o $d/k.cubin $src
    nvdisasm -hex $d/k.cubin > $d/k.sass
    cuobjdump -res-usage $d/k.cubin > $d/res-usage.txt
  }
  for d in /o/testset/*/; do
    echo "$(basename $d)" | grep -Eq "$ONLY" || { rm -rf $d; continue; }
    cp /w/perfmodel/probes/common.cuh /o/common.cuh
    sed -i "s|../../../probes/common.cuh|/o/common.cuh|" $d/harness.cu
    build $d $d/harness.cu || { echo "FAILED $(basename $d)" >> /o/testset-failed.txt; rm -rf $d; }
  done
  # round 6: mma_pipe also from the image compiler (host build above), as the test set is built
  if echo probe-mma_pipe | grep -Eq "$ONLY"; then
    mkdir -p /o/probe-mma_pipe_i
    build /o/probe-mma_pipe_i /w/perfmodel/probes/mma_pipe/probe.cu
  fi
  if echo probe-l2share_copy | grep -Eq "$ONLY"; then
    mkdir -p /o/probe-l2share_copy
    build /o/probe-l2share_copy /w/perfmodel/probes/l2share_copy/probe.cu
  fi
'
echo "built into $out"

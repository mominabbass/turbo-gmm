#!/bin/bash

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS="${CUVS_DEPS:-$(cd "$(dirname "$0")" && pwd)/.deps}"
GXX="${GXX:-/opt/rh/gcc-toolset-14/root/usr/bin/g++}"
NCCL="${NCCL_LIB:-/tmp/gmmgpu/.venv/lib/python3.12/site-packages/nvidia/nccl/lib}"

[ -d "$DEPS/libcuvs/include" ] || { echo "missing cuVS headers at $DEPS"; exit 1; }
[ -x "$GXX" ] || { echo "need gcc>=12 at $GXX"; exit 1; }

INC="-I$DEPS/libcuvs/include -I$DEPS/libraft/include -I$DEPS/librmm/include \
     -I$DEPS/libraft/include/rapids -I$DEPS/rapids_logger/include -I/usr/local/cuda/include"
LIBDIRS="-L$DEPS/libcuvs/lib64 -L$DEPS/librmm/lib64 -L$DEPS/libkvikio/lib64 \
         -L$DEPS/rapids_logger/lib64 -L$NCCL -L/usr/local/cuda/lib64"
RPATHS="-Wl,-rpath,$DEPS/libcuvs/lib64 -Wl,-rpath,$DEPS/librmm/lib64 \
        -Wl,-rpath,$DEPS/libkvikio/lib64 -Wl,-rpath,$DEPS/rapids_logger/lib64 \
        -Wl,-rpath,$NCCL -Wl,-rpath,$DEPS/../nightly/libcuvs_cu12.libs \
        -Wl,-rpath,/usr/local/cuda/lib64"

$GXX -std=c++20 -O2 -x c++ -c "$HERE/harness_spec.cu" -o "$HERE/harness_spec.o" $INC
$GXX -o "$HERE/harness_spec" "$HERE/harness_spec.o" $LIBDIRS \
     -lcuvs -lrmm -lkvikio -lrapids_logger -lcudart -lcublas -lcusolver $RPATHS
rm -f "$HERE/harness_spec.o"
echo "built: $HERE/harness_spec"

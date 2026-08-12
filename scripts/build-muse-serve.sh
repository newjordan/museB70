#!/usr/bin/env bash
# Build the muse-serve union worktree: lx/reorder-multicol-mkl head (9272b77c9)
# + Muse Glimmer loader patch (staged; from upstream 62bf73d25, via
# patches/muse-loader-on-c7d3bfe6d.patch) + the Q4_K small-N serving kernel.
# Mirrors the lx-reorder-multicol serving build's cmake flags.
set -euo pipefail

SRC_WT=${SRC_WT:-/home/frosty40/turbo/worktrees/muse-serve}
BUILD_DIR="$SRC_WT/build"

__old="$(set +o)"; set +eu
# shellcheck disable=SC1091
source /opt/intel/oneapi/setvars.sh --force >/dev/null 2>&1 || true
eval "$__old"; unset __old

grep -q MUSE_GLIMMER "$SRC_WT/src/llama-arch.cpp" || { echo "FATAL: muse patch not present" >&2; exit 1; }

cmake -S "$SRC_WT" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/opt/intel/oneapi/compiler/2026.0/bin/icx \
  -DCMAKE_CXX_COMPILER=/opt/intel/oneapi/compiler/2026.0/bin/icpx \
  -DGGML_SYCL=ON -DGGML_SYCL_TARGET=INTEL -DGGML_SYCL_F16=ON \
  -DGGML_SYCL_DNN=ON -DGGML_SYCL_GRAPH=ON -DGGML_SYCL_SUPPORT_LEVEL_ZERO_API=ON \
  -DGGML_NATIVE=ON -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_SERVER=ON \
  -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_MTMD_TOOL=ON \
  >"$BUILD_DIR-configure.log" 2>&1 || { echo "configure failed:"; tail -30 "$BUILD_DIR-configure.log"; exit 1; }

cmake --build "$BUILD_DIR" -j"${BUILD_JOBS:-24}" --target \
  llama-server llama-bench llama-batched-bench llama-perplexity llama-cli test-backend-ops \
  >"$BUILD_DIR-build.log" 2>&1 || { echo "build failed:"; tail -60 "$BUILD_DIR-build.log"; exit 1; }

echo "OK: $(ls -la "$BUILD_DIR/bin/llama-server" | awk '{print $5, $9}')"
echo "so_sha=$(sha256sum "$(readlink -f "$BUILD_DIR/bin/libggml-sycl.so.0")" | cut -c1-8)"

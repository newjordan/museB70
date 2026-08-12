#!/usr/bin/env bash
# Public Arc serving launcher for Muse Glimmer 30B Q4_K_M.
# Validated on Intel Arc Pro B70 (oneAPI 2026.0). Dense 28B — every token
# rereads ~15.6 GB of weights, so this is not a Laguna/Qwen-MoE speed class.
#
# B70 headline numbers need the museB70 SYCL build:
#   https://github.com/newjordan/museB70/releases/tag/v2026.08.12-b70
# Stock llama.cpp SYCL will load the model; the GGML_SYCL_LX_* knobs are no-ops there.
#
# Official weights (receipted file):
#   hf download meta-models/Muse-Glimmer-30B-GGUF muse-glimmer-30B-kquant-17gb.gguf
#   sha256 7e9b74b7c8875e9e265695df9613bf6290f2392e479ce740495a129019c488d8
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
LLAMA_BIN=${LLAMA_BIN:-llama-server}
PORT=${SERVE_PORT:-8095}
HOST=${SERVE_HOST:-127.0.0.1}
NPARALLEL=${SERVE_NP:-1}
CTX=${SERVE_CTX:-131072}
TEMPLATE=${TEMPLATE:-$HERE/muse-glimmer.jinja}

if [[ -n "${MODEL:-}" ]]; then
  :
elif [[ -f $HERE/muse-glimmer-30B-kquant-17gb.gguf ]]; then
  MODEL=$HERE/muse-glimmer-30B-kquant-17gb.gguf
elif [[ -f $HERE/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf ]]; then
  MODEL=$HERE/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf
else
  echo "FATAL: set MODEL= or put muse-glimmer-30B-kquant-17gb.gguf next to this script" >&2
  exit 2
fi
[[ -f $MODEL ]] || { echo "FATAL: missing model $MODEL" >&2; exit 2; }

export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:gpu}"
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
export GGML_SYCL_DISABLE_DNN=1
export GGML_SYCL_DISABLE_GRAPH=1
export GGML_SYCL_FUSE_NORM_ROPE=1
export GGML_SYCL_DISABLE_MOE_DUAL_DOWN=1
export GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN=1
export GGML_SYCL_DISABLE_QKV_SHARED_QUANT=1
# Load-bearing on the museB70 binary. Harmless no-ops on stock llama.cpp.
export GGML_SYCL_LX_REORDER_MULTICOL_MKL=1
export GGML_SYCL_LX_FATTN_PARALLEL_BLOCKS=24
export GGML_SYCL_LX_FATTN_DECODE_MASK_BOUNDS=1
export GGML_SYCL_LX_Q4_K_MMVQ_NCOLS_HOIST=1

JINJA=(--jinja)
[[ -f $TEMPLATE ]] && JINJA+=(--chat-template-file "$TEMPLATE")

exec "$LLAMA_BIN" \
  -m "$MODEL" \
  --alias muse-glimmer-30b-q4 \
  -a muse-glimmer-30b-q4 \
  -ngl 99 -fa on -ctk f16 -ctv f16 \
  -c "$CTX" -np "$NPARALLEL" \
  -b 4096 -ub 4096 \
  --swa-full \
  --temp 1.0 --top-k 64 --top-p 0.95 \
  -n -1 \
  --host "$HOST" --port "$PORT" \
  --metrics \
  --no-webui \
  "${JINJA[@]}"

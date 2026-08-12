#!/usr/bin/env bash
# Serve Muse Glimmer 30B (dense 28B, Q4_K_M) on the Arc Pro B70 at the FULL
# 131072-token trained context on the validated muse-serve build.
# Source commit: 3ce44d373. Promoted libggml-sycl SHA-256: aa90882dc06d653c.
#
# Conventions follow lx/package/laguna-b70-stack-20260812/serve-laguna.sh:
#   - Overrides are SERVE_CTX / SERVE_NP / SERVE_PORT (never bare CTX/PORT —
#     env.sh exports CTX=8192 and bare names silently clamp; 2026-08-10 trap).
#   - -c is the TOTAL KV pool split across -np slots. Muse KV is tiny
#     (2 KV heads + sliding-window 2048 on 39/52 layers): full 131072 ctx at
#     f16 costs only ~2 GiB, so f16 KV at full context is free on 30.3 GiB.
#   - Loader-resolution assert before exec.
#
# Model facts (gguf): 52 layers, ctx_train 131072, GQA 32/2, head 128,
# interleaved SWA window 2048, final-logit softcap 20. Official sampling
# (Meta model card, Aug 2026): temp 1.0, top-p 0.95, top-k 64.
#
# Bench receipts (results/sweep-20260811, champ-muse-probe binary):
#   pp512 933 / pp4096 1322-1336 / tg128 28.6 (bandwidth wall: 15.6 GB weights)
#   tg128@d8192 27.0, @d32768 25.2. oneDNN: -0.5% pp (keep native GEMM).
#   SYCL graph: flat (keep off). ub1024: -11% pp4096 (no).
# See RESULTS.md for the union-build depth A/B that pinned -ub/-b and LX knobs.

# Never let an inherited xtrace setting print API keys or key-file contents.
set +x
export -n SHELLOPTS 2>/dev/null || true
set -euo pipefail
umask 077
PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export PATH

die() { echo "FATAL: $*" >&2; exit 2; }
MODE=serve
if (( $# == 1 )) && [[ "$1" == --check ]]; then
  MODE=check
elif (( $# == 1 )) && [[ "$1" == --verify-model ]]; then
  MODE=verify-model
elif (( $# != 0 )); then
  die "positional llama-server arguments are disabled; use validated SERVE_* settings or --check"
fi

# Serve from a read-only release snapshot, never from the mutable CMake output.
# SERVE_RELEASE_ROOT exists for checksum-failure regression tests and can only
# select a byte-identical copy because the manifest digest is fixed here.
RELEASE_ROOT=${SERVE_RELEASE_ROOT:-/home/frosty40/turbo/muse/releases/muse-serve-3ce44d373}
RELEASE_MANIFEST=$RELEASE_ROOT/SHA256SUMS
EXPECTED_MANIFEST_SHA256=53e20fe1250b7c2bcaf5dc4753b0ac80ed62469e58656758570436124dbd71e5
BIN=$RELEASE_ROOT/bin
MODEL=/mnt/data2tb/benchmodels/muse-glimmer-30B-kquant-17gb.gguf
EXPECTED_MODEL_SHA256=7e9b74b7c8875e9e265695df9613bf6290f2392e479ce740495a129019c488d8
EXPECTED_MODEL_STAT=66308:45219851:16756681056:1786468158:1786468158
PORT=${SERVE_PORT:-8095}
HOST=${SERVE_HOST:-127.0.0.1}
CORS_ORIGINS=${SERVE_CORS_ORIGINS:-localhost}
NPARALLEL=${SERVE_NP:-1}       # 1 = full 131072 ctx per request
CTX=${SERVE_CTX:-131072}
UB=${SERVE_UB:-4096}
BB=${SERVE_BB:-4096}
LOG=${SERVE_LOG:-journal}

# llama.cpp consumes LLAMA_ARG_* before CLI parsing. Clear all ambient llama
# options and loader injection, then populate only the validated auth inputs.
while IFS= read -r var; do unset "$var"; done < <(compgen -A variable LLAMA_ARG_ || true)
while IFS= read -r var; do unset "$var"; done < <(compgen -A variable GGML_SYCL_ || true)
unset LLAMA_API_KEY LD_PRELOAD LD_AUDIT
unset AIP_MODE AIP_HTTP_PORT AIP_HEALTH_ROUTE AIP_PREDICT_ROUTE

[[ -x "$BIN/llama-server" ]] || die "missing executable $BIN/llama-server"
[[ -f "$MODEL" && -r "$MODEL" && ! -L "$MODEL" ]] || die "missing regular model $MODEL"
[[ "$PORT" =~ ^[0-9]+$ ]] && (( PORT > 0 && PORT < 65536 )) || die "SERVE_PORT must be 1..65535"
[[ "$NPARALLEL" =~ ^[1-9][0-9]*$ ]] || die "SERVE_NP must be positive"
[[ "$CTX" =~ ^[1-9][0-9]*$ ]] || die "SERVE_CTX must be positive"
[[ "$UB" =~ ^[1-9][0-9]*$ ]] || die "SERVE_UB must be positive"
[[ "$BB" =~ ^[1-9][0-9]*$ ]] || die "SERVE_BB must be positive"
(( BB >= UB )) || die "SERVE_BB must be greater than or equal to SERVE_UB"
[[ "${SERVE_SWA_FULL:-1}" == 0 || "${SERVE_SWA_FULL:-1}" == 1 ]] || die "SERVE_SWA_FULL must be 0 or 1"
[[ "${SERVE_WEBUI:-0}" == 0 || "${SERVE_WEBUI:-0}" == 1 ]] || die "SERVE_WEBUI must be 0 or 1"
[[ "${SERVE_ALLOW_INSECURE_NETWORK:-0}" == 0 || "${SERVE_ALLOW_INSECURE_NETWORK:-0}" == 1 ]] || die "SERVE_ALLOW_INSECURE_NETWORK must be 0 or 1"
[[ "${SERVE_GPU_LOCK_WAIT:-0}" =~ ^[0-9]+$ ]] || die "SERVE_GPU_LOCK_WAIT must be a non-negative integer"
[[ ! "$HOST" =~ [[:cntrl:]] ]] || die "SERVE_HOST contains control characters"
[[ ! "$CORS_ORIGINS" =~ [[:cntrl:]] ]] || die "SERVE_CORS_ORIGINS contains control characters"
[[ "$CORS_ORIGINS" != "*" ]] || die "wildcard CORS is disabled"
[[ "$LOG" == journal || "$LOG" == /* ]] || die "SERVE_LOG must be an absolute path or journal"
[[ "$NPARALLEL" == 1 && "$CTX" == 131072 && "$UB" == 4096 && "$BB" == 4096 ]] || \
  die "validated single-seat profile requires SERVE_NP=1 SERVE_CTX=131072 SERVE_UB=4096 SERVE_BB=4096"
[[ "${SERVE_SWA_FULL:-1}" == 1 ]] || die "validated max-context profile requires SERVE_SWA_FULL=1"
[[ "${SERVE_WEBUI:-0}" == 0 ]] || die "validated API-only profile requires SERVE_WEBUI=0"
if [[ -n "${SERVE_API_KEY_FILE:-}" ]]; then
  [[ "$SERVE_API_KEY_FILE" == /* ]] || die "SERVE_API_KEY_FILE must be an absolute path"
  [[ -f "$SERVE_API_KEY_FILE" && -r "$SERVE_API_KEY_FILE" && ! -L "$SERVE_API_KEY_FILE" ]] || \
    die "SERVE_API_KEY_FILE must be a readable regular file, not a symlink: $SERVE_API_KEY_FILE"
  KEY_MODE=$(stat -c '%a' "$SERVE_API_KEY_FILE")
  (( (8#$KEY_MODE & 077) == 0 )) || die "SERVE_API_KEY_FILE must not be group/world accessible: mode $KEY_MODE"
  KEY_UID=$(stat -c '%u' "$SERVE_API_KEY_FILE")
  [[ "$KEY_UID" == "$EUID" ]] || die "SERVE_API_KEY_FILE must be owned by uid $EUID"
  awk 'NF && $1 !~ /^#/ { found = 1; exit } END { exit !found }' "$SERVE_API_KEY_FILE" || \
    die "SERVE_API_KEY_FILE contains no usable keys"
fi
[[ -z "${SERVE_API_KEY:-}" || -z "${SERVE_API_KEY_FILE:-}" ]] || die "set only one of SERVE_API_KEY and SERVE_API_KEY_FILE"
if command -v ss >/dev/null 2>&1 && ss -H -ltn "sport = :$PORT" 2>/dev/null | grep -q .; then
  die "TCP port $PORT is already listening"
fi
[[ -n "${SERVE_API_KEY:-}" ]] && export LLAMA_API_KEY="$SERVE_API_KEY"
[[ -n "${SERVE_API_KEY_FILE:-}" ]] && export LLAMA_ARG_API_KEY_FILE="$SERVE_API_KEY_FILE"
AUTH_ENABLED=0
[[ -n "${SERVE_API_KEY:-}" || -n "${SERVE_API_KEY_FILE:-}" ]] && AUTH_ENABLED=1
# --swa-full keeps old SWA KV so edited turns reuse cache to the divergence
# point. It costs about 5.1 GiB at 131072. The decode bounds kernel skips the
# masked old prefix while preserving that cache history. Compact/ring SWA is
# an experimental alternate mode, not part of this validated ship launcher.
SWA_FULL_FLAG=(--swa-full)
WEBUI_FLAG=(--no-webui)

# NB: do NOT `source setvars.sh` here — it exits the shell under `set -euo pipefail`.
ONEAPI=/opt/intel/oneapi
export LD_LIBRARY_PATH="$BIN:$ONEAPI/tcm/1.5/lib:$ONEAPI/umf/1.1/lib:$ONEAPI/tbb/2023.0/env/../lib/intel64/gcc4.8:$ONEAPI/mpi/2021.18/opt/mpi/libfabric/lib:$ONEAPI/mpi/2021.18/lib:$ONEAPI/mkl/2026.0/lib:$ONEAPI/ippcp/2026.0/lib/:$ONEAPI/ipp/2026.0/lib:$ONEAPI/dnnl/2026.0/lib:$ONEAPI/debugger/2026.0/opt/debugger/lib:$ONEAPI/dal/2026.0/lib:$ONEAPI/compiler/2026.0/opt/compiler/lib:$ONEAPI/compiler/2026.0/lib:$ONEAPI/ccl/2022.0/lib"

# Ship env — measured for Muse on 2026-08-11/12 (see RESULTS.md), base block
# mirrors the validated champion serving env (kill-switches included: the
# broken MoE paths never fire on a dense model, but keeping them killed is
# free and keeps one env story across the rig).
export ONEAPI_DEVICE_SELECTOR=level_zero:gpu
export ZE_AFFINITY_MASK=0
export GGML_SYCL_DISABLE_DNN=1          # A/B 2026-08-11: oneDNN -0.5% pp on Muse
export GGML_SYCL_DISABLE_GRAPH=1        # A/B 2026-08-11: flat; keep compat default
export GGML_SYCL_FUSE_NORM_ROPE=1
export GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE=0
export GGML_SYCL_DISABLE_MOE_DUAL_DOWN=1
export GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN=1
export GGML_SYCL_DISABLE_QKV_SHARED_QUANT=1
# lx stack knobs — muse A/Bs 2026-08-12 (RESULTS.md):
#   REORDER_MULTICOL_MKL=1 is LOAD-BEARING for serving: after the first decode
#     reorders the weights, wide prefill batches fall into 8-col reorder-MMVQ
#     at ~61 t/s. This fall-through restores dense GEMM: server prefill
#     61 -> 985-1207 t/s (16x). llama-bench CANNOT see this (its pp tests run
#     before any decode) — do not re-tune this knob with llama-bench.
#   FATTN_PARALLEL_BLOCKS=24: best one-seat full-depth split width.
#   FATTN_DECODE_MASK_BOUNDS=1: skip fully masked SWA prefix tiles at decode.
#   EXPERT_TILE_GEMM: MoE-only, dormant on dense — off.
export GGML_SYCL_LX_REORDER_MULTICOL_MKL=1
export GGML_SYCL_LX_FATTN_PARALLEL_BLOCKS=24
export GGML_SYCL_LX_FATTN_DECODE_MASK_BOUNDS=1
# Reuse each reordered Q4_K weight/scale load across the active decode slots.
# 2026-08-12 warmed knee: +7.8% / +8.5% / +8.2% aggregate TG at np=2/4/8.
export GGML_SYCL_LX_Q4_K_MMVQ_NCOLS_HOIST=1

# Validate the fixed manifest before inspecting its executables. The release
# directories and files must not be group/world writable, preventing an
# accidental rebuild or cooperative-user write from racing startup.
[[ -f "$RELEASE_MANIFEST" && ! -L "$RELEASE_MANIFEST" ]] || {
  echo "FATAL: missing regular release manifest $RELEASE_MANIFEST" >&2
  exit 3
}
MANIFEST_SHA256=$(sha256sum -- "$RELEASE_MANIFEST" | awk '{print $1}')
[[ "$MANIFEST_SHA256" == "$EXPECTED_MANIFEST_SHA256" ]] || {
  echo "FATAL: release manifest SHA-256 $MANIFEST_SHA256 is not validated" >&2
  exit 3
}
for release_path in "$RELEASE_ROOT" "$BIN" "$RELEASE_MANIFEST"; do
  RELEASE_MODE=$(stat -c '%a' "$release_path")
  (( (8#$RELEASE_MODE & 022) == 0 )) || {
    echo "FATAL: release path is group/world writable: $release_path mode=$RELEASE_MODE" >&2
    exit 3
  }
done
while read -r release_link release_target; do
  [[ -L "$BIN/$release_link" && "$(readlink "$BIN/$release_link")" == "$release_target" ]] || {
    echo "FATAL: release symlink changed: $release_link" >&2
    exit 3
  }
done <<'EOF'
libggml-base.so libggml-base.so.0
libggml-base.so.0 libggml-base.so.0.17.0
libggml-cpu.so libggml-cpu.so.0
libggml-cpu.so.0 libggml-cpu.so.0.17.0
libggml-sycl.so libggml-sycl.so.0
libggml-sycl.so.0 libggml-sycl.so.0.17.0
libggml.so libggml.so.0
libggml.so.0 libggml.so.0.17.0
libllama-common.so libllama-common.so.0
libllama-common.so.0 libllama-common.so.0.0.10178
libllama.so libllama.so.0
libllama.so.0 libllama.so.0.0.10178
libmtmd.so libmtmd.so.0
libmtmd.so.0 libmtmd.so.0.0.10178
EOF
while read -r _ release_file; do
  RELEASE_FILE_MODE=$(stat -Lc '%a' "$RELEASE_ROOT/$release_file") || {
    echo "FATAL: missing release artifact $release_file" >&2
    exit 3
  }
  (( (8#$RELEASE_FILE_MODE & 022) == 0 )) || {
    echo "FATAL: release artifact is group/world writable: $release_file mode=$RELEASE_FILE_MODE" >&2
    exit 3
  }
done <"$RELEASE_MANIFEST"
if ! (cd "$RELEASE_ROOT" && sha256sum --check --strict --quiet SHA256SUMS); then
  echo "FATAL: release artifact checksum mismatch" >&2
  exit 3
fi

# Loader-resolution assert (FINDING_20260810 trap family). Every local
# llama/ggml DSO must come from the immutable release snapshot.
LDD_OUTPUT=$(ldd "$BIN/llama-server" 2>&1) || {
  echo "FATAL: ldd failed for release llama-server" >&2
  exit 3
}
if grep -q 'not found' <<<"$LDD_OUTPUT"; then
  echo "FATAL: release llama-server has unresolved libraries" >&2
  grep 'not found' <<<"$LDD_OUTPUT" >&2
  exit 3
fi
while read -r soname arrow resolved _; do
  case "$soname" in
    libllama*|libggml*|libmtmd*)
      [[ "$arrow" == "=>" && "$resolved" == "$BIN"/* ]] || {
        echo "FATAL: $soname resolves from $resolved, not $BIN" >&2
        exit 3
      }
      ;;
  esac
done <<<"$LDD_OUTPUT"
RESOLVED_SO=$(awk '/libggml-sycl/ {print $3; exit}' <<<"$LDD_OUTPUT")
if [[ -z "$RESOLVED_SO" ]]; then
  echo "FATAL: llama-server did not resolve libggml-sycl" >&2
  exit 3
fi
if [[ "$RESOLVED_SO" != "$BIN"/* ]]; then
  echo "FATAL: llama-server resolves libggml-sycl from $RESOLVED_SO, not $BIN" >&2
  exit 3
fi
SERVER_SHA256=$(sha256sum -- "$BIN/llama-server" | awk '{print $1}')
SYCL_SHA256=$(sha256sum -- "$(readlink -f "$RESOLVED_SO")" | awk '{print $1}')
MODEL_STAT=$(stat -Lc '%d:%i:%s:%Y:%Z' "$MODEL")
if [[ "$MODEL_STAT" != "$EXPECTED_MODEL_STAT" ]]; then
  echo "FATAL: model identity changed: $MODEL_STAT (expected $EXPECTED_MODEL_STAT)" >&2
  exit 3
fi
MODEL_VERIFICATION=identity
if [[ "$MODE" == verify-model ]]; then
  MODEL_SHA256=$(sha256sum -- "$MODEL" | awk '{print $1}')
  [[ "$MODEL_SHA256" == "$EXPECTED_MODEL_SHA256" ]] || {
    echo "FATAL: model SHA-256 $MODEL_SHA256 is not validated" >&2
    exit 3
  }
  MODEL_VERIFICATION=sha256
fi
if [[ "$HOST" != "127.0.0.1" && "$HOST" != "localhost" && "$HOST" != "::1" ]]; then
  [[ "${SERVE_ALLOW_INSECURE_NETWORK:-0}" == 1 ]] || \
    die "refusing direct network bind on $HOST; use localhost behind TLS or set the explicit insecure-network break glass"
  [[ -n "${SERVE_API_KEY_FILE:-}" ]] || \
    die "direct network bind requires a private SERVE_API_KEY_FILE; inline/env keys are disabled"
  echo "WARNING: direct $HOST serving is plain HTTP; use only behind trusted transport" >&2
fi
if [[ "$MODE" != serve ]]; then
  echo "Muse serving validation PASS mode=$MODE release_sha=$MANIFEST_SHA256 server_sha=$SERVER_SHA256 sycl_sha=$SYCL_SHA256 model_sha=$EXPECTED_MODEL_SHA256 model_verify=$MODEL_VERIFICATION model_stat=$MODEL_STAT host=$HOST auth=$AUTH_ENABLED webui=${SERVE_WEBUI:-0} cors=$CORS_ORIGINS pb=$GGML_SYCL_LX_FATTN_PARALLEL_BLOCKS mask_bounds=$GGML_SYCL_LX_FATTN_DECODE_MASK_BOUNDS sanitized_env=1"
  exit 0
fi
SERVER_PID=
stop_server() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 120); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 0.25
    done
    if kill -0 "$SERVER_PID" 2>/dev/null; then
      kill -KILL "$SERVER_PID" 2>/dev/null || true
    fi
  fi
  if [[ -n "$SERVER_PID" ]]; then
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=
  fi
}
cleanup() {
  trap - EXIT INT TERM
  stop_server
  lx_gpu_lock_leave
}

# Hold the canonical fleet lock for the complete server lifetime. Clear every
# library escape hatch and inherited function before sourcing the helper.
unset LX_GPU_LOCK_SKIP LX_GPU_ALLOW_BUSY LX_GPU_LOCK LX_GPU_LOCK_META LX_ROOT LX_GPU_LOCK_FD
unset _LX_GPU_LOCK_LOADED _LX_GPU_LOCK_HELD
unset -f lx_gpu_lock_enter lx_gpu_lock_leave lx_gpu_foreign_procs 2>/dev/null || true
export LX_ROOT=/home/frosty40/turbo/lx
export LX_GPU_LOCK=/home/frosty40/turbo/lx/results/.b70-gpu.lock
export LX_GPU_LOCK_META=/home/frosty40/turbo/lx/results/.b70-gpu.lock.meta
export LX_GPU_LOCK_FD=9
export LX_GPU_LOCK_WAIT=${SERVE_GPU_LOCK_WAIT:-0}
# shellcheck disable=SC1091
source /home/frosty40/turbo/lx/scripts/lib-gpu-lock.sh
lx_gpu_lock_enter "muse-serve-$PORT" || exit $?
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

CONFIG_LINE="[serve-muse] $(date -u +%FT%TZ) release=$RELEASE_ROOT manifest_sha=${MANIFEST_SHA256:0:8} server_sha=${SERVER_SHA256:0:8} so_sha=${SYCL_SHA256:0:8} model_sha=${EXPECTED_MODEL_SHA256:0:8} ctx=$CTX np=$NPARALLEL b=$BB ub=$UB swa_full=${SERVE_SWA_FULL:-1} pb=$GGML_SYCL_LX_FATTN_PARALLEL_BLOCKS mask_bounds=$GGML_SYCL_LX_FATTN_DECODE_MASK_BOUNDS host=$HOST port=$PORT auth=$AUTH_ENABLED webui=${SERVE_WEBUI:-0} cors=$CORS_ORIGINS"
if [[ "$LOG" == journal ]]; then
  echo "$CONFIG_LINE" >&2
else
  if [[ -L "$LOG" || ( -e "$LOG" && ! -f "$LOG" ) ]]; then
    die "SERVE_LOG must be a regular file, not a symlink: $LOG"
  fi
  mkdir -p "$(dirname "$LOG")" || die "cannot create SERVE_LOG directory"
  touch "$LOG" || die "cannot create SERVE_LOG $LOG"
  chmod 600 "$LOG" || die "cannot make SERVE_LOG private"
  echo "$CONFIG_LINE" >>"$LOG" || die "cannot write SERVE_LOG $LOG"
fi

SERVER_CMD=(
  "$BIN/llama-server"
  -m "$MODEL"
  --alias muse-glimmer-30b-q4
  -a muse-glimmer-30b-q4
  -ngl 99 -fa on -ctk f16 -ctv f16
  -c "$CTX" -np "$NPARALLEL"
  -b "$BB" -ub "$UB"
  -t 16
  --host "$HOST" --port "$PORT"
  --cors-origins "$CORS_ORIGINS"
  --jinja
  --temp 1.0 --top-k 64 --top-p 0.95
  -n -1
  --metrics
  "${WEBUI_FLAG[@]}"
  "${SWA_FULL_FLAG[@]}"
)
if [[ "$LOG" == journal ]]; then
  "${SERVER_CMD[@]}" &
else
  "${SERVER_CMD[@]}" >>"$LOG" 2>&1 &
fi
SERVER_PID=$!

SERVER_RC=0
wait "$SERVER_PID" || SERVER_RC=$?
SERVER_PID=
cleanup
exit "$SERVER_RC"

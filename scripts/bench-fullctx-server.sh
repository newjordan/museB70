#!/usr/bin/env bash
# Canonical Muse B70 gate: real llama-server, one seat, live 129k-token depth.
#
# Runs fresh process-static OFF-A / ON / OFF-B arms under one exclusive GPU
# lock. The Python client first evaluates a frozen 129024-token WikiText prompt
# with cache disabled, then sends that same token prompt with cache enabled.
# llama-server deliberately re-evaluates the final prompt token, proving the
# timed completion starts at cache_n=129023 / prompt_n=1 without truncation.
#
# No slot save/restore is used: serialized slot state can omit the masked SWA
# history and is not an honest substitute for a live --swa-full context.
# Run directly after building the candidate binary; candidate env/value knobs
# below can be overridden for a different process-static kernel experiment.
set -euo pipefail

ROOT=/home/frosty40/turbo/muse
REPO=${REPO:-/home/frosty40/turbo/worktrees/muse-serve}
BIN=${BENCH_BIN:-$REPO/build/bin}
MODEL=${MODEL:-/mnt/data2tb/benchmodels/muse-glimmer-30B-kquant-17gb.gguf}
SOURCE=${FULLCTX_SOURCE:-/home/frosty40/data/wikitext-2-raw/wiki.train.raw}
OUT=${FULLCTX_OUT:-$ROOT/results/fullctx-server-$(date -u +%Y%m%dT%H%M%SZ)}
FIXTURE=${FULLCTX_FIXTURE:-$ROOT/results/fixtures/muse-wikitext-train-129024.tokens.json}
PORT=${FULLCTX_PORT:-18097}
TOKEN_COUNT=129024
N_CTX=131072
N_PREDICT=${FULLCTX_N_PREDICT:-256}
QUALITY_N_PREDICT=${FULLCTX_QUALITY_N_PREDICT:-32}
N_PROBS=${FULLCTX_N_PROBS:-128}
MAX_OFF_SPREAD_PCT=${FULLCTX_MAX_OFF_SPREAD_PCT:-2.0}
MIN_ON_IMPROVEMENT_PCT=${FULLCTX_MIN_ON_IMPROVEMENT_PCT:-1.0}
SPEED_NOISE_MULTIPLIER=${FULLCTX_SPEED_NOISE_MULTIPLIER:-2.0}
PROB_NOISE_MULTIPLIER=${FULLCTX_PROB_NOISE_MULTIPLIER:-2.0}
CANDIDATE_ENV_NAME=${CANDIDATE_ENV_NAME:-GGML_SYCL_LX_FATTN_DECODE_MASK_BOUNDS}
CANDIDATE_OFF_VALUE=${CANDIDATE_OFF_VALUE:-0}
CANDIDATE_ON_VALUE=${CANDIDATE_ON_VALUE:-1}
ARITHMETIC_EXPECTED=${ARITHMETIC_EXPECTED:-1}
PYTHON=${PYTHON:-python3}
CLIENT=$ROOT/scripts/fullctx-server-bench.py
ONEAPI=/opt/intel/oneapi
ONEAPI_LIBS="$ONEAPI/tcm/1.5/lib:$ONEAPI/umf/1.1/lib:$ONEAPI/tbb/2023.0/env/../lib/intel64/gcc4.8:$ONEAPI/mpi/2021.18/opt/mpi/libfabric/lib:$ONEAPI/mpi/2021.18/lib:$ONEAPI/mkl/2026.0/lib:$ONEAPI/ippcp/2026.0/lib/:$ONEAPI/ipp/2026.0/lib:$ONEAPI/dnnl/2026.0/lib:$ONEAPI/debugger/2026.0/opt/debugger/lib:$ONEAPI/dal/2026.0/lib:$ONEAPI/compiler/2026.0/opt/compiler/lib:$ONEAPI/compiler/2026.0/lib:$ONEAPI/ccl/2022.0/lib"

# shellcheck disable=SC1091
source /home/frosty40/turbo/lx/scripts/lib-gpu-lock.sh

server_pid=
cleanup() {
    stop_server
    lx_gpu_lock_leave
}

die() {
    echo "FATAL: $*" >&2
    exit 1
}

assert_loader() {
    local resolved
    resolved=$(LD_LIBRARY_PATH="$BIN:$ONEAPI_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        ldd "$BIN/llama-server" 2>/dev/null | awk '/libggml-sycl/ { print $3; exit }')
    [[ -n "$resolved" ]] || die "llama-server did not resolve libggml-sycl"
    [[ "$resolved" == "$BIN"/* ]] || die "llama-server resolves libggml-sycl from $resolved, not $BIN"
}

assert_no_listener() {
    if command -v ss >/dev/null 2>&1 && ss -H -ltn "sport = :$PORT" 2>/dev/null | grep -q .; then
        die "TCP port $PORT is already listening"
    fi
}

start_server() {
    local arm=$1
    local candidate_value=$2
    local log=$OUT/$arm-server.log

    [[ -z "$server_pid" ]] || die "refusing to start $arm while PID $server_pid is tracked"
    assert_no_listener
    assert_loader

    env \
        LD_LIBRARY_PATH="$BIN:$ONEAPI_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
        ZE_AFFINITY_MASK=0 \
        GGML_SYCL_DISABLE_DNN=1 \
        GGML_SYCL_DISABLE_GRAPH=1 \
        GGML_SYCL_FUSE_NORM_ROPE=1 \
        GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE=0 \
        GGML_SYCL_DISABLE_MOE_DUAL_DOWN=1 \
        GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN=1 \
        GGML_SYCL_DISABLE_QKV_SHARED_QUANT=1 \
        GGML_SYCL_LX_REORDER_MULTICOL_MKL=1 \
        GGML_SYCL_LX_FATTN_PARALLEL_BLOCKS=24 \
        GGML_SYCL_LX_Q4_K_MMVQ_NCOLS_HOIST=1 \
        "$CANDIDATE_ENV_NAME=$candidate_value" \
        "$BIN/llama-server" \
            -m "$MODEL" --alias muse-fullctx-gate \
            -ngl 99 -fa on -ctk f16 -ctv f16 \
            -c "$N_CTX" -np 1 -b 4096 -ub 4096 -t 16 \
            --swa-full \
            --host 127.0.0.1 --port "$PORT" \
            --temp 1.0 --top-k 64 --top-p 0.95 \
            -n -1 --metrics \
            >"$log" 2>&1 &
    server_pid=$!

    local ready=0
    for _ in $(seq 1 600); do
        if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            ready=1
            break
        fi
        if ! kill -0 "$server_pid" 2>/dev/null; then
            wait "$server_pid" || true
            server_pid=
            tail -n 80 "$log" >&2 || true
            die "$arm server died during model load"
        fi
        sleep 0.5
    done
    if (( ready == 0 )); then
        tail -n 80 "$log" >&2 || true
        die "$arm server did not become healthy within 300 seconds"
    fi

    grep -Fq 'n_ctx_slot = 131072' "$log" || die "$arm log does not prove n_ctx_slot=131072"
    grep -Eq 'n_slots = 1|n_parallel = 1' "$log" || die "$arm log does not prove one slot"
    grep -Fq 'using full-size SWA cache' "$log" || die "$arm log does not prove --swa-full"
}

stop_server() {
    if [[ -n "$server_pid" ]]; then
        if kill -0 "$server_pid" 2>/dev/null; then
            kill -TERM "$server_pid" 2>/dev/null || true
            for _ in $(seq 1 120); do
                kill -0 "$server_pid" 2>/dev/null || break
                sleep 0.25
            done
            if kill -0 "$server_pid" 2>/dev/null; then
                local tracked_exe
                tracked_exe=$(readlink -f "/proc/$server_pid/exe" 2>/dev/null || true)
                if [[ "$tracked_exe" == "$(readlink -f "$BIN/llama-server")" ]]; then
                    echo "WARNING: tracked server PID $server_pid did not exit after SIGTERM; sending SIGKILL" >&2
                    kill -KILL "$server_pid" 2>/dev/null || true
                else
                    echo "WARNING: PID $server_pid no longer resolves to the tracked llama-server; not signaling it" >&2
                fi
            fi
        fi
        wait "$server_pid" || true
        server_pid=
    fi
}

run_fixture_server() {
    if [[ -f "$FIXTURE" ]]; then
        "$PYTHON" - "$FIXTURE" "$TOKEN_COUNT" "$SOURCE" "$MODEL" "$MODEL_SHA256" <<'PY'
import hashlib
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
count = int(sys.argv[2])
source = pathlib.Path(sys.argv[3])
model = pathlib.Path(sys.argv[4])
model_sha256 = sys.argv[5]
fixture = json.loads(path.read_text(encoding="utf-8"))
tokens = fixture.get("tokens")
if not isinstance(tokens, list) or len(tokens) != count:
    raise SystemExit(f"invalid token count in {path}")
payload = (json.dumps(tokens, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
expected = {
    "schema": 2,
    "model": str(model),
    "model_sha256": model_sha256,
    "source": str(source),
    "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "tokenize_add_special": True,
    "tokenize_parse_special": True,
    "token_count": count,
    "first_token_id": tokens[0],
    "last_token_id": tokens[-1],
    "tokens_sha256": hashlib.sha256(payload).hexdigest(),
}
wrong = {key: (fixture.get(key), value) for key, value in expected.items() if fixture.get(key) != value}
if wrong:
    raise SystemExit(f"incompatible fixture {path}: {wrong!r}")
PY
        return
    fi

    start_server fixture "$CANDIDATE_OFF_VALUE"
    "$PYTHON" "$CLIENT" fixture \
        --base-url "http://127.0.0.1:$PORT" \
        --source "$SOURCE" --model "$MODEL" \
        --model-sha256 "$MODEL_SHA256" \
        --fixture "$FIXTURE" --out-dir "$OUT" \
        --token-count "$TOKEN_COUNT"
    stop_server
}

run_arm() {
    local arm=$1
    local candidate_value=$2
    start_server "$arm" "$candidate_value"
    "$PYTHON" "$CLIENT" request \
        --base-url "http://127.0.0.1:$PORT" \
        --arm "$arm" --fixture "$FIXTURE" --out-dir "$OUT" \
        --token-count "$TOKEN_COUNT" --n-ctx "$N_CTX" \
        --n-predict "$N_PREDICT" \
        --quality-n-predict "$QUALITY_N_PREDICT" --n-probs "$N_PROBS"
    stop_server
}

[[ -x "$BIN/llama-server" ]] || die "missing executable $BIN/llama-server"
[[ -r "$MODEL" ]] || die "missing model $MODEL"
[[ -r "$SOURCE" ]] || die "missing WikiText source $SOURCE"
[[ -f "$CLIENT" ]] || die "missing client $CLIENT"
[[ "$PORT" =~ ^[0-9]+$ ]] || die "FULLCTX_PORT must be an integer"
(( PORT > 0 && PORT < 65536 )) || die "FULLCTX_PORT must be between 1 and 65535"
[[ "$N_PREDICT" =~ ^[1-9][0-9]*$ ]] || die "FULLCTX_N_PREDICT must be positive"
[[ "$QUALITY_N_PREDICT" =~ ^[1-9][0-9]*$ ]] || die "FULLCTX_QUALITY_N_PREDICT must be positive"
[[ "$N_PROBS" =~ ^[1-9][0-9]*$ ]] || die "FULLCTX_N_PROBS must be positive"
[[ "$MAX_OFF_SPREAD_PCT" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "FULLCTX_MAX_OFF_SPREAD_PCT must be non-negative"
[[ "$MIN_ON_IMPROVEMENT_PCT" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "FULLCTX_MIN_ON_IMPROVEMENT_PCT must be non-negative"
[[ "$SPEED_NOISE_MULTIPLIER" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "FULLCTX_SPEED_NOISE_MULTIPLIER must be non-negative"
[[ "$PROB_NOISE_MULTIPLIER" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "FULLCTX_PROB_NOISE_MULTIPLIER must be non-negative"
[[ "$CANDIDATE_ENV_NAME" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || die "invalid CANDIDATE_ENV_NAME"
[[ "$ARITHMETIC_EXPECTED" == 0 || "$ARITHMETIC_EXPECTED" == 1 ]] || die "ARITHMETIC_EXPECTED must be 0 or 1"
MODEL_SHA256=$(sha256sum "$MODEL" | awk '{print $1}')

lx_gpu_lock_enter "muse-fullctx-server" || exit $?
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

[[ ! -e "$OUT" ]] || die "output directory already exists: $OUT"
mkdir -p "$OUT" "$(dirname "$FIXTURE")"

assert_loader
run_fixture_server
mapfile -t fixture_meta < <("$PYTHON" - "$FIXTURE" <<'PY'
import json
import sys

fixture = json.load(open(sys.argv[1], encoding="utf-8"))
print(fixture["tokens_sha256"])
print(fixture["first_token_id"])
print(fixture["last_token_id"])
PY
)
{
    echo "timestamp_utc=$(date -u +%FT%TZ)"
    echo "bin=$BIN"
    echo "server_sha256=$(sha256sum "$BIN/llama-server" | awk '{print $1}')"
    echo "sycl_sha256=$(sha256sum "$(readlink -f "$BIN/libggml-sycl.so.0")" | awk '{print $1}')"
    echo "repo_head=$(git -C "$REPO" rev-parse HEAD)"
    echo "model=$MODEL"
    echo "model_size=$(stat -c %s "$MODEL")"
    echo "model_sha256=$MODEL_SHA256"
    echo "fixture=$FIXTURE"
    echo "fixture_sha256=$(sha256sum "$FIXTURE" | awk '{print $1}')"
    echo "fixture_tokens_sha256=${fixture_meta[0]}"
    echo "fixture_first_token_id=${fixture_meta[1]}"
    echo "fixture_last_token_id=${fixture_meta[2]}"
    echo "tokenize_add_special=true"
    echo "tokenize_parse_special=true"
    echo "candidate_env_name=$CANDIDATE_ENV_NAME"
    echo "candidate_off_value=$CANDIDATE_OFF_VALUE"
    echo "candidate_on_value=$CANDIDATE_ON_VALUE"
    echo "arithmetic_expected=$ARITHMETIC_EXPECTED"
    echo "ctx=$N_CTX"
    echo "slots=1"
    echo "token_count=$TOKEN_COUNT"
    echo "n_predict=$N_PREDICT"
    echo "quality_n_predict=$QUALITY_N_PREDICT"
    echo "quality_n_probs=$N_PROBS"
    echo "ship_sampling=temp1_topk64_topp0.95_minp0"
    echo "quality_sampling=greedy_temp0_topn_probs"
    echo "max_off_spread_pct=$MAX_OFF_SPREAD_PCT"
    echo "min_on_improvement_pct=$MIN_ON_IMPROVEMENT_PCT"
    echo "speed_noise_multiplier=$SPEED_NOISE_MULTIPLIER"
    echo "prob_noise_multiplier=$PROB_NOISE_MULTIPLIER"
    echo "kv=f16/f16"
    echo "swa_full=1"
} >"$OUT/manifest.txt"

run_arm off-a "$CANDIDATE_OFF_VALUE"
run_arm on "$CANDIDATE_ON_VALUE"
run_arm off-b "$CANDIDATE_OFF_VALUE"

compare_args=()
(( ARITHMETIC_EXPECTED == 1 )) && compare_args+=(--arithmetic-expected)
"$PYTHON" "$CLIENT" compare \
    --out-dir "$OUT" --token-count "$TOKEN_COUNT" --n-ctx "$N_CTX" \
    --max-off-spread-pct "$MAX_OFF_SPREAD_PCT" \
    --min-on-improvement-pct "$MIN_ON_IMPROVEMENT_PCT" \
    --speed-noise-multiplier "$SPEED_NOISE_MULTIPLIER" \
    --prob-noise-multiplier "$PROB_NOISE_MULTIPLIER" \
    "${compare_args[@]}"

echo "full-context server receipts -> $OUT"

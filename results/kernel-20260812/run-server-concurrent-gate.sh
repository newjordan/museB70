#!/usr/bin/env bash
set -euo pipefail

source /home/frosty40/turbo/lx/scripts/lib-gpu-lock.sh
lx_gpu_lock_enter "muse-kernel-server-gate" || exit $?

OUT=/home/frosty40/turbo/muse/results/kernel-20260812/server-gate
HARNESS=/home/frosty40/turbo/muse/results/kernel-20260812/server-concurrent-gate.py
PROMPT=/home/frosty40/turbo/lx/correctness/wikitext-23k.txt
MODEL=/mnt/data2tb/benchmodels/muse-glimmer-30B-kquant-17gb.gguf
CONTROL=/home/frosty40/turbo/worktrees/muse-serve/build/bin
CANDIDATE=/home/frosty40/turbo/worktrees/muse-serve/build-smalln/bin
ONEAPI=/opt/intel/oneapi
LIBS="$ONEAPI/tcm/1.5/lib:$ONEAPI/umf/1.1/lib:$ONEAPI/tbb/2023.0/env/../lib/intel64/gcc4.8:$ONEAPI/mpi/2021.18/opt/mpi/libfabric/lib:$ONEAPI/mpi/2021.18/lib:$ONEAPI/mkl/2026.0/lib:$ONEAPI/ippcp/2026.0/lib/:$ONEAPI/ipp/2026.0/lib:$ONEAPI/dnnl/2026.0/lib:$ONEAPI/debugger/2026.0/opt/debugger/lib:$ONEAPI/dal/2026.0/lib:$ONEAPI/compiler/2026.0/opt/compiler/lib:$ONEAPI/compiler/2026.0/lib:$ONEAPI/ccl/2022.0/lib"

mkdir -p "$OUT"
server_pid=

cleanup() {
    if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
        kill -TERM "$server_pid"
        wait "$server_pid" || true
    fi
    lx_gpu_lock_leave
}
trap cleanup EXIT INT TERM

assert_loader() {
    local bin=$1
    local resolved
    resolved=$(LD_LIBRARY_PATH="$bin:$LIBS" ldd "$bin/llama-server" | awk '/libggml-sycl/ { print $3; exit }')
    if [[ "$resolved" != "$bin"/* ]]; then
        echo "FATAL: $bin/llama-server resolves libggml-sycl from $resolved" >&2
        exit 3
    fi
}

start_server() {
    local arm=$1
    local bin=$2
    local port=$3
    local hoist=$4

    assert_loader "$bin"
    env \
        LD_LIBRARY_PATH="$bin:$LIBS" \
        ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
        ZE_AFFINITY_MASK=0 \
        GGML_SYCL_DISABLE_GRAPH=1 \
        GGML_SYCL_DISABLE_DNN=1 \
        GGML_SYCL_FUSE_NORM_ROPE=1 \
        GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE=0 \
        GGML_SYCL_DISABLE_MOE_DUAL_DOWN=1 \
        GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN=1 \
        GGML_SYCL_DISABLE_QKV_SHARED_QUANT=1 \
        GGML_SYCL_LX_REORDER_MULTICOL_MKL=1 \
        GGML_SYCL_LX_FATTN_PARALLEL_BLOCKS=32 \
        GGML_SYCL_LX_Q4_K_MMVQ_NCOLS_HOIST="$hoist" \
        "$bin/llama-server" \
            -m "$MODEL" --alias muse-kernel-gate \
            -ngl 99 -fa on -ctk f16 -ctv f16 \
            -c 16384 -np 8 -b 4096 -ub 4096 -t 16 \
            --host 127.0.0.1 --port "$port" \
            --temp 1.0 --top-k 64 --top-p 0.95 -n -1 --metrics \
            >"$OUT/$arm-server.log" 2>&1 &
    server_pid=$!
}

stop_server() {
    if [[ -n "$server_pid" ]]; then
        kill -TERM "$server_pid"
        wait "$server_pid" || true
        server_pid=
    fi
}

run_wave() {
    local arm=$1
    local port=$2
    local concurrency=$3
    local ready_flag=${4:-}

    python3 "$HARNESS" \
        --base-url "http://127.0.0.1:$port" \
        --arm "$arm" --out-dir "$OUT" \
        --prompt-file "$PROMPT" --prompt-chars 6000 \
        --concurrency "$concurrency" --n-predict 64 \
        $ready_flag
}

start_server control "$CONTROL" 18095 0
run_wave control 18095 8 --wait-ready
stop_server

start_server candidate "$CANDIDATE" 18096 1
run_wave candidate 18096 8 --wait-ready
for concurrency in 2 3 4 5 6 7; do
    run_wave "candidate-n$concurrency" 18096 "$concurrency"
done
stop_server

{
    for slot in $(seq 0 7); do
        cmp "$OUT/control-slot$slot.tokens.json" "$OUT/candidate-slot$slot.tokens.json"
        cmp "$OUT/control-slot$slot.content.txt" "$OUT/candidate-slot$slot.content.txt"
        printf 'slot=%d tokens=identical content=identical\n' "$slot"
    done
    printf 'server_log_dispatches:\n'
    grep -F '[lx-q4-k-ncols-hoist]' "$OUT/candidate-server.log"
} | tee "$OUT/compare.txt"

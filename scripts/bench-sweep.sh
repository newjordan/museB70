#!/usr/bin/env bash
# Muse Glimmer 30B on B70 — serving-config knob sweep.
# Pattern copied from lx/scripts/ab-crossmodel.sh (lock + loader assert + JSON receipts).
# Usage: bench-sweep.sh <leg-name> [ENV=VAL ...] -- <llama-bench extra flags>
set -euo pipefail

LEG=${1:?usage: bench-sweep.sh <leg> [ENV=VAL ...] -- <bench flags>}
shift
ENVS=()
while [[ $# -gt 0 && "$1" != "--" ]]; do ENVS+=("$1"); shift; done
[[ $# -gt 0 ]] && shift   # drop --

BIN=${BIN:-/home/frosty40/turbo/worktrees/champ-muse-probe/build/bin}
MODEL=${MODEL:-/mnt/data2tb/benchmodels/muse-glimmer-30B-kquant-17gb.gguf}
OUT=${OUT:-/home/frosty40/turbo/muse/results/sweep-20260811}
LX_ROOT=/home/frosty40/turbo/lx
mkdir -p "$OUT"

ONEAPI=/opt/intel/oneapi
ONEAPI_LIBS="$ONEAPI/tcm/1.5/lib:$ONEAPI/umf/1.1/lib:$ONEAPI/tbb/2023.0/env/../lib/intel64/gcc4.8:$ONEAPI/mpi/2021.18/opt/mpi/libfabric/lib:$ONEAPI/mpi/2021.18/lib:$ONEAPI/mkl/2026.0/lib:$ONEAPI/ippcp/2026.0/lib/:$ONEAPI/ipp/2026.0/lib:$ONEAPI/dnnl/2026.0/lib:$ONEAPI/debugger/2026.0/opt/debugger/lib:$ONEAPI/dal/2026.0/lib:$ONEAPI/compiler/2026.0/opt/compiler/lib:$ONEAPI/compiler/2026.0/lib:$ONEAPI/ccl/2022.0/lib"

# shellcheck disable=SC1091
source "$LX_ROOT/scripts/lib-gpu-lock.sh"
lx_gpu_lock_enter "muse-sweep:$LEG" || exit $?
trap 'lx_gpu_lock_leave' EXIT

resolved="$(env LD_LIBRARY_PATH="$BIN:$ONEAPI_LIBS" ldd "$BIN/llama-bench" | awk '/libggml-sycl/ {print $3; exit}')"
if [[ "$resolved" != "$BIN"/* ]]; then
  echo "FATAL: resolves libggml-sycl from $resolved, not $BIN" >&2
  exit 3
fi
echo "[$LEG] so_sha=$(sha256sum "$(readlink -f "$BIN/libggml-sycl.so.0")" | cut -c1-8) $(date -u +%T) env: ${ENVS[*]:-none}"

# Base env = validated champion ship block (mirrors serve-laguna.sh / crossmodel champ leg).
# Per-leg ENVS come last so they override.
env LD_LIBRARY_PATH="$BIN:$ONEAPI_LIBS" \
    ONEAPI_DEVICE_SELECTOR=level_zero:gpu ZE_AFFINITY_MASK=0 \
    GGML_SYCL_DISABLE_GRAPH=1 GGML_SYCL_DISABLE_DNN=1 \
    GGML_SYCL_FUSE_NORM_ROPE=1 \
    GGML_SYCL_DISABLE_MUL_MAT_ADD_FUSE=0 \
    GGML_SYCL_DISABLE_MOE_DUAL_DOWN=1 \
    GGML_SYCL_DISABLE_MOE_DUAL_MULTITOKEN=1 \
    GGML_SYCL_DISABLE_QKV_SHARED_QUANT=1 \
    "${ENVS[@]}" \
  "$BIN/llama-bench" \
    -m "$MODEL" -ngl 99 --n-cpu-moe 0 \
    --split-mode layer --main-gpu 0 --tensor-split 0 --device auto \
    -t 16 --cpu-mask 0x0 --cpu-strict 0 \
    -ctk f16 -ctv f16 \
    --no-kv-offload 0 --no-op-offload 0 --no-host 0 \
    --prio 0 --load-mode mmap --poll 50 --delay 0 \
    "$@" \
    -o json >"$OUT/$LEG.json" 2>"$OUT/$LEG.log"

python3 - "$OUT/$LEG.json" "$LEG" <<'PYEOF'
import json,sys,statistics
rows=json.load(open(sys.argv[1]))
for r in rows:
    kind=f"pp{r['n_prompt']}" if r['n_gen']==0 else f"tg{r['n_gen']}"
    if r.get('n_depth'): kind+=f"@d{r['n_depth']}"
    ts=r['samples_ts']
    sd=statistics.stdev(ts) if len(ts)>1 else 0.0
    print(f"  {sys.argv[2]:14s} {kind:12s} {statistics.mean(ts):9.2f} t/s  (sd {sd:5.2f}, n={len(ts)})")
PYEOF

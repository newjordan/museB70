#!/usr/bin/env bash
# No-GPU regression tests for serve-muse.sh safety and operator preflight.
set -euo pipefail

MUSE=/home/frosty40/turbo/muse
SERVE=$MUSE/serve-muse.sh
TMP=$(mktemp -d /tmp/muse-serve-preflight.XXXXXX)

cleanup() {
    if [[ -n "${PORT_PID:-}" ]] && kill -0 "$PORT_PID" 2>/dev/null; then
        kill -TERM "$PORT_PID" 2>/dev/null || true
        wait "$PORT_PID" 2>/dev/null || true
    fi
    if [[ -n "${HOLDER_PID:-}" ]] && kill -0 "$HOLDER_PID" 2>/dev/null; then
        kill -TERM "$HOLDER_PID" 2>/dev/null || true
        wait "$HOLDER_PID" 2>/dev/null || true
    fi
    chmod -R u+w "$TMP" 2>/dev/null || true
    rm -rf "$TMP"
}
trap cleanup EXIT INT TERM

expect_failure() {
    local label=$1 expected_rc=$2 pattern=$3
    shift 3
    local out rc failure_log=$TMP/failure-$label.log
    set +e
    out=$(env SERVE_LOG="$failure_log" "$@" bash "$SERVE" --check 2>&1)
    rc=$?
    set -e
    [[ "$rc" == "$expected_rc" ]] || { echo "$label: rc=$rc, expected $expected_rc" >&2; return 1; }
    grep -q "$pattern" <<<"$out" || { echo "$label: missing error pattern $pattern" >&2; return 1; }
    ! grep -q 'ACQUIRED' <<<"$out" || { echo "$label: acquired GPU lock before failing" >&2; return 1; }
    [[ ! -e "$failure_log" ]] || { echo "$label: created serving log before failing" >&2; return 1; }
    echo "PASS $label"
}

expect_launch_failure() {
    local label=$1 expected_rc=$2 pattern=$3
    shift 3
    local out rc
    set +e
    out=$(env "$@" bash "$SERVE" 2>&1)
    rc=$?
    set -e
    [[ "$rc" == "$expected_rc" ]] || { echo "$label: rc=$rc, expected $expected_rc" >&2; return 1; }
    grep -q "$pattern" <<<"$out" || { echo "$label: missing error pattern $pattern" >&2; return 1; }
    printf '%s' "$out"
}

bash -n "$SERVE" "$MUSE/scripts/smoke-test.sh" "$MUSE/scripts/bench-fullctx-server.sh"
expect_failure invalid-port 2 'SERVE_PORT must be' SERVE_PORT=0
expect_failure invalid-batch 2 'SERVE_BB must be greater' SERVE_UB=4096 SERVE_BB=2048
expect_failure invalid-swa 2 'SERVE_SWA_FULL must be' SERVE_SWA_FULL=bad
expect_failure nonship-context 2 'validated single-seat profile requires' SERVE_CTX=8192
expect_failure nonship-swa 2 'validated max-context profile requires' SERVE_SWA_FULL=0
expect_failure nonship-webui 2 'validated API-only profile requires' SERVE_WEBUI=1
expect_failure direct-network 2 'refusing direct network bind' SERVE_HOST=0.0.0.0 SERVE_PORT=18101
expect_failure inline-network-key 2 'requires a private SERVE_API_KEY_FILE' SERVE_HOST=0.0.0.0 SERVE_PORT=18101 SERVE_ALLOW_INSECURE_NETWORK=1 SERVE_API_KEY=inline-secret
expect_failure wildcard-cors 2 'wildcard CORS is disabled' SERVE_CORS_ORIGINS='*'

set +e
positional_out=$(bash "$SERVE" --host 0.0.0.0 2>&1)
positional_rc=$?
set -e
[[ "$positional_rc" == 2 ]]
grep -q 'positional llama-server arguments are disabled' <<<"$positional_out"
! grep -q 'ACQUIRED' <<<"$positional_out"
echo 'PASS protected-positional-arguments'

# A byte change in any release executable/DSO must fail the fixed manifest.
cp -a --reflink=auto "$MUSE/releases/muse-serve-3ce44d373" "$TMP/drift-release"
chmod u+w "$TMP/drift-release/bin/libllama-common.so.0.0.10178"
printf 'drift' >>"$TMP/drift-release/bin/libllama-common.so.0.0.10178"
expect_failure release-dso-drift 3 'release artifact checksum mismatch' \
    SERVE_RELEASE_ROOT="$TMP/drift-release" SERVE_PORT=18101
cp -a --reflink=auto "$MUSE/releases/muse-serve-3ce44d373" "$TMP/link-release"
chmod u+w "$TMP/link-release/bin"
ln -sfn libggml-cpu.so.0.17.0 "$TMP/link-release/bin/libggml-sycl.so.0"
expect_failure release-link-drift 3 'release symlink changed' \
    SERVE_RELEASE_ROOT="$TMP/link-release" SERVE_PORT=18101

printf '%s\n' 'public-key' >"$TMP/public.key"
chmod 644 "$TMP/public.key"
expect_failure public-key-file 2 'must not be group/world accessible' SERVE_API_KEY_FILE="$TMP/public.key"

printf '%s\n' '# comment only' >"$TMP/empty.key"
chmod 600 "$TMP/empty.key"
expect_failure empty-key-file 2 'contains no usable keys' SERVE_API_KEY_FILE="$TMP/empty.key"

ln -s "$TMP/empty.key" "$TMP/symlink.key"
expect_failure symlink-key-file 2 'not a symlink' SERVE_API_KEY_FILE="$TMP/symlink.key"

printf '%s\n' 'private-key' >"$TMP/private.key"
chmod 600 "$TMP/private.key"
secure_err=$TMP/secure.err
SERVE_HOST=0.0.0.0 SERVE_ALLOW_INSECURE_NETWORK=1 SERVE_API_KEY_FILE="$TMP/private.key" \
    SERVE_PORT=18101 GGML_SYCL_LX_FATTN_PARALLEL_BLOCKS=999 \
    GGML_SYCL_LX_FATTN_DECODE_MASK_BOUNDS=0 AIP_MODE=PREDICTION AIP_HTTP_PORT=9999 \
    bash "$SERVE" --check >"$TMP/secure.out" 2>"$secure_err"
grep -q 'plain HTTP' "$secure_err"
grep -q 'host=0.0.0.0 auth=1 webui=0 cors=localhost pb=24 mask_bounds=1 sanitized_env=1' "$TMP/secure.out"
! grep -R -q 'private-key' "$TMP/secure.out" "$secure_err"
echo 'PASS secure-network-preflight'

set +e
xtrace_out=$(env SHELLOPTS=xtrace SERVE_API_KEY=xtrace-secret SERVE_PORT=18101 \
    bash "$SERVE" --check 2>&1)
xtrace_rc=$?
set -e
[[ "$xtrace_rc" == 0 ]]
! grep -q 'xtrace-secret' <<<"$xtrace_out"
echo 'PASS xtrace-secret-redaction'

# Ambient llama options cannot smuggle auth or agent/tool behavior into the
# validated child configuration.
expect_failure ambient-auth-bypass 2 'refusing direct network bind' \
    SERVE_HOST=0.0.0.0 SERVE_PORT=18101 LLAMA_ARG_API_KEY_FILE="$TMP/private.key" LLAMA_ARG_AGENT=1

python3 -m http.server 18099 --bind 127.0.0.1 >"$TMP/http.log" 2>&1 &
PORT_PID=$!
for _ in $(seq 1 100); do
    ss -H -ltn 'sport = :18099' 2>/dev/null | grep -q . && break
    sleep 0.05
done
ss -H -ltn 'sport = :18099' 2>/dev/null | grep -q .
expect_failure port-collision 2 'TCP port 18099 is already listening' SERVE_PORT=18099
kill -TERM "$PORT_PID"
wait "$PORT_PID" 2>/dev/null || true
PORT_PID=

# A held fleet lock must reject a second launcher before it starts a server,
# even if the caller supplies every library bypass variable.
ready=$TMP/holder.ready
(
    set -euo pipefail
    source /home/frosty40/turbo/lx/scripts/lib-gpu-lock.sh
    lx_gpu_lock_enter serve-preflight-holder
    touch "$ready"
    trap 'lx_gpu_lock_leave' EXIT
    while [[ -e "$ready" ]]; do sleep 0.05; done
) &
HOLDER_PID=$!
for _ in $(seq 1 100); do [[ -e "$ready" ]] && break; sleep 0.05; done
[[ -e "$ready" ]]
lock_out=$(expect_launch_failure lock-collision 75 'BUSY: another job holds' \
    SERVE_PORT=18099 SERVE_LOG="$TMP/collision.log" \
    LX_GPU_LOCK_SKIP=1 LX_GPU_ALLOW_BUSY=1 LX_GPU_LOCK="$TMP/private.lock" LX_GPU_LOCK_META="$TMP/private.meta")
! grep -q 'exclusive lock disabled' <<<"$lock_out"
[[ ! -e "$TMP/collision.log" ]]
echo 'PASS lock-bypass-sanitized'
rm -f "$ready"
wait "$HOLDER_PID"
HOLDER_PID=
test ! -e /home/frosty40/turbo/lx/results/.b70-gpu.lock.meta
echo 'PASS lock-release-clean'

# A post-acquisition filesystem failure must still clear metadata and release
# the flock; this path never starts llama-server or touches the GPU.
touch "$TMP/not-a-directory"
log_failure_out=$(expect_launch_failure log-setup-cleanup 2 'cannot create SERVE_LOG directory' \
    SERVE_PORT=18099 SERVE_LOG="$TMP/not-a-directory/server.log")
grep -q 'ACQUIRED' <<<"$log_failure_out"
grep -q 'RELEASED' <<<"$log_failure_out"
test ! -e /home/frosty40/turbo/lx/results/.b70-gpu.lock.meta
echo 'PASS post-lock-failure-cleanup'

ln -s "$TMP/missing-log-target" "$TMP/dangling.log"
log_link_out=$(expect_launch_failure dangling-log-symlink 2 'SERVE_LOG must be a regular file, not a symlink' \
    SERVE_PORT=18099 SERVE_LOG="$TMP/dangling.log")
grep -q 'ACQUIRED' <<<"$log_link_out"
grep -q 'RELEASED' <<<"$log_link_out"
[[ ! -e "$TMP/missing-log-target" ]]
test ! -e /home/frosty40/turbo/lx/results/.b70-gpu.lock.meta
echo 'PASS dangling-log-symlink-cleanup'

echo 'serve preflight regression suite: PASS'

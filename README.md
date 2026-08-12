<p align="center">
  <img src="assets/brand/muse-b70-hero.png" alt="Muse B70 — neoclassical marble Muse in a tech-noir compute sanctuary" width="100%">
</p>

# muse — Muse Glimmer 30B serving on the Arc Pro B70

Serving config project for **Muse Glimmer 30B** (Meta, day-0 release 2026-08-11):
dense 28B, Q4_K_M (`/mnt/data2tb/benchmodels/muse-glimmer-30B-kquant-17gb.gguf`,
15.6 GB), served at the **full 131072-token trained context** on the B70.

<table>
  <tr>
    <td width="31%"><img src="assets/brand/muse-b70-emblem.png" alt="Muse B70 marble-and-bronze emblem"></td>
    <td width="69%">
      <h3>One seat. Full context. Verified output.</h3>
      <p><strong>19.032 t/s</strong> sampled server decode at 129,024 cached tokens.<br>
      <strong>503.19 t/s</strong> full-context prime.<br>
      <strong>+79.681%</strong> full-SWA decode from the shipped mask-bounds kernel.<br>
      <strong>Exact</strong> greedy token and content parity against both controls.<br>
      <strong>0.000494 KLD / 99.240% same-top</strong> broad quality gate.</p>
    </td>
  </tr>
</table>

## Quick start

```bash
bash /home/frosty40/turbo/muse/serve-muse.sh --check          # fast, no GPU/model hash
bash /home/frosty40/turbo/muse/serve-muse.sh --verify-model   # adds full model SHA-256
bash /home/frosty40/turbo/muse/serve-muse.sh          # -> http://127.0.0.1:8095/v1
curl -s localhost:8095/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"muse-glimmer-30b-q4","messages":[{"role":"user","content":"hi"}]}'
```

The private GitHub release carries the checksum-pinned runtime separately
from Git history. Restore it after cloning:

```bash
cd /home/frosty40/turbo/muse
gh release download v2026.08.12-b70 --repo newjordan/museB70 \
  --pattern 'muse-serve-3ce44d373-linux-b70.tar.zst'
sha256sum -c releases/ASSET_SHA256SUMS
tar --zstd -C releases -xf muse-serve-3ce44d373-linux-b70.tar.zst
unlink muse-serve-3ce44d373-linux-b70.tar.zst
bash serve-muse.sh --verify-model
```

The model is intentionally never uploaded. Kernel source provenance and
importable patches are in `KERNEL_PROVENANCE.md` and `patches/kernel/`.

Systemd: `muse-b70.service` (template in this dir, NOT installed by default).
It keeps the listener on localhost, sends output to journald, and limits
failure restarts to three attempts per five minutes.
No-GPU regression suites: `bash scripts/test-serve-preflight.sh` and
`python3 scripts/test-fullctx-server-compare.py`.

Muse thinks in a `to=self` channel: responses carry `reasoning_content`
alongside `content` (the specialized parser from upstream 62bf73d25 is
patched into the muse-serve build — `patches/muse-chat-parser-common.patch`;
without it channel headers leak into `content`). Hydra surface: `turbo:8095`
(candidate).

Client contract: when extending a conversation, echo both the assistant's
`reasoning_content` and `content` fields into the assistant history message.
Reasoning consumes output budget, so use an adequate `max_tokens`; the
canonical two-turn gate uses 128.

## Production serving contract

**One GPU client at a time.** Concurrent Level-Zero clients wedge the xe
driver (lx/env.sh). `serve-muse.sh` holds the fleet lock for the complete
server lifetime and refuses to start beside Laguna, treebeard, or a GPU job.
It deletes ambient lock bypass variables/functions, restores the canonical
fleet lock paths, and releases both flock and metadata on normal exit and
post-lock startup failure.

The production launcher is intentionally not a general llama-server wrapper:

- It fixes the validated profile at `-np 1 -c 131072 -ub 4096 -b 4096`,
  f16 KV, full-SWA, and API-only operation. Attempts to change NP, context,
  batch geometry, SWA mode, or Web UI fail closed. Those experiments belong
  in a separate benchmark/manual launcher, never `serve-muse.sh`.
- Positional llama-server arguments are rejected. Before setting the exact
  ship knobs it clears every ambient `LLAMA_ARG_*` and `GGML_SYCL_*` variable,
  plus `LLAMA_API_KEY`, `LD_PRELOAD`, and `LD_AUDIT`.
- It serves from the read-only snapshot
  `releases/muse-serve-3ce44d373`, not mutable CMake output. Startup pins the
  SHA256SUMS file itself and then verifies the server and every local
  llama/ggml DSO in that manifest; loader resolution must stay inside the
  release directory.
- `--check` validates release hashes, DSO resolution, fixed configuration,
  network policy, and the model's stat identity without starting a server or
  hashing 15.6 GB. `--verify-model` additionally checks the complete model
  SHA-256. Neither mode acquires the GPU lock.

Production listens only on localhost. Expose it remotely through a TLS
reverse proxy or an SSH tunnel while keeping `SERVE_HOST=127.0.0.1`. Direct
LAN plain HTTP is break-glass only: it requires both
`SERVE_ALLOW_INSECURE_NETWORK=1` and an absolute, owner-controlled
`SERVE_API_KEY_FILE`; inline/environment keys are rejected for a non-loopback
bind. The key file must be a non-symlink regular file, inaccessible to group
and world, and contain a usable key. Wildcard CORS is rejected. The default
log target is quota-managed journald; an explicit file must be absolute,
private, and non-symlink.

## Model facts (from GGUF)

| fact | value | serving consequence |
|---|---|---|
| arch | `muse-glimmer` (dense 28B, 52 layers) | needs the Muse loader patch — see below |
| ctx_train | 131072 | serve `-c 131072 -np 1` |
| GQA | 32 Q / **2 KV** heads, dim 128 | KV ≈ 52 KB/token at f16 |
| attention | interleaved SWA, window 2048 (39/52 layers) | compact/ring KV is ~2 GiB; full-SWA retains old KV for branch reuse and costs ~5.1 GiB more |
| logits | final softcap 20, scale 0.196 | no attention softcap → flash attention fine |
| template | embedded (harmony-style "Onyx ATEM") | `--jinja`; extracted fallback: `muse-glimmer.jinja` |
| sampling | temp 1.0, top-p 0.95, top-k 64 (official card) | baked into serve-muse.sh |

VRAM at the fixed one-seat full-SWA ship config is ~25 GiB of 30.3. Compact
ring-cache experiments require a separate manual/benchmark launcher; the
production launcher does not expose that mode.

## Release and binary

Production executes only from the read-only release snapshot
`/home/frosty40/turbo/muse/releases/muse-serve-3ce44d373`. Its fixed
`SHA256SUMS` manifest has SHA-256
`53e20fe1250b7c2bcaf5dc4753b0ac80ed62469e58656758570436124dbd71e5`
and covers `llama-server` plus every bundled local llama/ggml DSO. Validated
artifact identities are:

- llama-server SHA-256:
  `e5bfa43f97d5f4de00bc5ec51c73722a56e0c6e2ad544fd61b3f7c801201fa86`
- libggml-sycl SHA-256:
  `aa90882dc06d653c3f7a580ff6e1f34a5d84e02225eebcbf92cf243c0db69f06`

The fixed model SHA-256 is
`7e9b74b7c8875e9e265695df9613bf6290f2392e479ce740495a129019c488d8`;
its startup stat identity is
`66308:45219851:16756681056:1786468158:1786468158`
(`device:inode:size:mtime:ctime`). The source worktree is branch
`lx/muse-serve` at commit `3ce44d373`; it is provenance, not the serving
path. Its gated `build-smalln` counterpart differs in build-directory
metadata; the linked executable code, data, and SYCL offload sections are
identical.
The stock serving build (`lx-reorder-multicol`)
cannot load Muse; the probe worktrees (`{base,champ}-muse-probe`) can but lack
the lx stack knobs.

## Live release validation

`results/serve-release-history-quality-20260812T1752Z.md` is the canonical
live smoke for the immutable release. Its server log records manifest
`53e20fe1`, server `e5bfa43f`, SYCL `aa90882d`, model `7e9b74b7`, the fixed
one-seat/full-SWA configuration, localhost, authentication enabled, Web UI
disabled, and local CORS. Runtime checks passed exactly: unauthenticated chat
returned HTTP 401, an untrusted browser origin was rejected, localhost origin
was allowed, the authenticated root returned HTTP 404, chat returned
`MUSE-OK` then `SECOND-OK`.
An authenticated `/apply-template` gate also rendered old marker
`MUSE-OLD-REASONING-741` and recent marker `MUSE-RECENT-REASONING-852`
exactly once each, in order, from assistant `reasoning_content` history. This
smoke then took the actual turn-one response and echoed both its
`reasoning_content` and `content` into the turn-two request; turn two returned
exact `SECOND-OK` and reused cache with `prompt_n=15`. Both requests used
`max_tokens=128`. This validates the client-managed reasoning-history path.
The separate `--reasoning-preserve` option remains disabled/no-op for Muse and
is not the mechanism under test.

The earlier `results/serve-final-quality-20260812T171634Z.md` smoke ran from
mutable `build/bin`; it did not record or validate the llama-server SHA and is
not the release-integrity receipt.

Both no-GPU regression suites pass. `scripts/test-serve-preflight.sh` covers
fixed-profile rejection, positional/environment bypasses, manifested-DSO
drift, key-file and network policy, port/lock collisions, canonical lock
cleanup, and post-lock failure cleanup. `scripts/test-fullctx-server-compare.py`
protects receipt provenance, OFF-control stability, speed, greedy quality,
probability drift, and source-receipt immutability.

## Measured (receipts in `results/`, RESULTS.md for the story)

![Full-context Muse B70 server benchmark: 19.032 tokens per second at 129024 cached tokens, 79.681 percent faster than bounds-off](assets/bench/fullctx-server.svg)

Every headline number above is tied to the frozen 129,024-token fixture and
the checked-in server receipts; the plot is not a synthetic projection.

Champion-binary sweep, 2026-08-11 (pp512/tg128 r5; depth legs r2-3):

| leg | pp512 | pp4096 | tg128 |
|---|---|---|---|
| control (DNN off, graph off, ub2048) | 933.1 | 1322.3 | 28.62 |
| oneDNN on | 927.7 | 1315.7 | 28.66 |
| ub4096 | 930.1 | **1336.1** | 28.64 |
| ub1024 | 925.8 | 1182.3 | 28.56 |
| SYCL graph on | 927.5 | — | 28.63 |
| depth: tg128@d8192 / @d32768 | | | 27.04 / 25.17 |
| depth: pp2048@d32768 | | 1045.9 | (tg64 25.37) |

Reading: decode sits at the 15.6 GB × 28.6 t/s ≈ 446 GB/s bandwidth wall —
kernel knobs don't move it at d0. Prefill-at-depth is healthy (SWA). oneDNN
and SYCL graph: keep off. Base-control binary ≈ champion on Muse at depth
too (1044.8 / 25.14) — no Laguna-calibration penalty.

Those depth rows use llama-bench's compact/ring SWA cache. They are not
`--swa-full` measurements.

Union-build (muse-serve) knobs, shipped ON:

- **`GGML_SYCL_LX_REORDER_MULTICOL_MKL=1` — load-bearing.** Real server
  prefill is 61 t/s without it (weights reorder for MMVQ on first decode,
  then wide prefill batches crawl); with it: **1277 ± 1.7 t/s steady-state**
  (20.9×; first request ~1026 with one-time warmup — canonical receipt
  results/canonical-pp.log). llama-bench cannot see this (pp tests run
  before any decode) — never re-tune this knob with llama-bench.
  Greedy-gated token-identical.
- `GGML_SYCL_LX_FATTN_PARALLEL_BLOCKS=24` is the best one-seat split width at
  full depth. The earlier width-32 choice was based on compact/ring d32K.
- `GGML_SYCL_LX_FATTN_DECODE_MASK_BOUNDS=1` derives live KV tile bounds from
  the mask and skips fully masked SWA prefix tiles during VEC decode while
  retaining the physical history needed for branch/edit cache reuse. B70
  uses a 1024-thread cooperative mask scanner; smaller devices retain the
  128-thread fallback.
- `GGML_SYCL_LX_Q4_K_MMVQ_NCOLS_HOIST=1` reuses each reordered Q4_K
  weight/scale load across the active decode columns. Warmed multi-slot TG
  improves by about 8% at `-np 2/4/8`, with flat `-np 1` and prefill.
  Exact Muse-shape ops passed 21/21; broad KLD passed at 0.000499 / 99.191%;
  the stable eight-slot greedy gate was byte- and token-identical in 8/8 slots.

Ship flags: `-ub 4096 -b 4096 -fa on -ctk f16 -ctv f16 -c 131072 -np 1
--swa-full` (multi-slot decode still scales only 1.59× aggregate at 8 slots —
so -np 1; see RESULTS.md knee table). `--swa-full` (+~5 GiB) makes edited /
regenerated turns reuse cache to the divergence point instead of full
re-prefill (642 vs 2664 tokens in the A/B). The production launcher fixes
this full-SWA profile.

![Muse B70 multi-slot scaling: aggregate throughput rises only 1.59 times from one to eight seats, supporting the single-seat production profile](assets/bench/single-seat-scaling.svg)

Server reality must be reported by cache mode:

| cache mode | 129K decode | behavior |
|---|---:|---|
| compact/ring (experimental/manual launch only) | ~19.7 t/s warmed llama-bench estimate | fastest linear-history mode; edits can require full re-prefill |
| full-SWA, bounds OFF | 10.592 t/s | retains branch history but scans the masked old prefix |
| full-SWA, bounds ON (ship) | **19.032 t/s** | same retained history, **+79.681%** vs OFF |

The final full-SWA run used a fresh server and the same frozen 129024-token
fixture as the OFF-A/OFF-B controls; prime was 256.46 s / 503.19 t/s and the
greedy sidecar reached 19.399 t/s. Its greedy tokens and content match both
controls exactly. The probability gate passes against measured OFF/OFF
noise: worst candidate mean TV proxy 0.006181 vs control 0.006233; max TV
0.018135 vs 0.016477; max selected-logprob delta 0.051246, below the 2×
control allowance 0.060169. Broad compile-regression KLD also passes at
0.000494 mean / 99.240% same-top; the full-depth path itself is covered by
the exact greedy and probability gates. Final receipt:
`results/fullctx-server-wide-final-20260812T163904Z/`.

The earlier 128-thread scanner's 17.292 t/s A/B/A remains in
`results/fullctx-server-final-20260812T1605Z/` as an intermediate validation,
not the ship headline. Officially sampled tails were not bit-stable even
between OFF controls, so the quality claim rests on exact greedy output,
noise-calibrated probabilities, and KLD rather than sampled coincidence.

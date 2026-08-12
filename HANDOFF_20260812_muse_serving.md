# HANDOFF 2026-08-12 — Muse Glimmer 30B serving on B70: config DONE, Q4_K small-N + full-SWA decode improved

For the next agent(s). Everything below is receipted; nothing is folklore.

## State: the serving config is finished and validated

- `serve-muse.sh` in this dir — port 8095, full 131072 ctx, `-np 1`, f16 KV,
  `--swa-full`, `-ub 4096 -b 4096 -fa on`, official sampling (temp 1.0 /
  top-p 0.95 / top-k 64), split width 24, and masked-prefix decode bounds.
  This is a fixed production profile: the launcher rejects changes to NP,
  context, batch geometry, full-SWA, or API-only mode, and accepts no
  positional llama-server passthrough. Experimental profiles require a
  separate benchmark/manual launcher.
- Production serves only from read-only snapshot
  `releases/muse-serve-3ce44d373`, never mutable CMake output. Its fixed
  SHA256SUMS manifest SHA-256 is
  `53e20fe1250b7c2bcaf5dc4753b0ac80ed62469e58656758570436124dbd71e5`
  and covers llama-server plus every bundled local llama/ggml DSO. The
  validated llama-server SHA-256 is
  `e5bfa43f97d5f4de00bc5ec51c73722a56e0c6e2ad544fd61b3f7c801201fa86`;
  validated libggml-sycl SHA-256 is
  `aa90882dc06d653c3f7a580ff6e1f34a5d84e02225eebcbf92cf243c0db69f06`.
  Loader resolution for every local DSO must remain inside the snapshot.
  Source provenance is worktree `/home/frosty40/turbo/worktrees/muse-serve`,
  branch `lx/muse-serve`, commit `3ce44d373`; it is not the serving path.
- Model identity is pinned at every startup by stat tuple
  `66308:45219851:16756681056:1786468158:1786468158`
  (`device:inode:size:mtime:ctime`). `serve-muse.sh --check` validates the
  fixed release/config/network policy and this stat identity without starting
  the server or hashing 15.6 GB. `serve-muse.sh --verify-model` additionally
  verifies model SHA-256
  `7e9b74b7c8875e9e265695df9613bf6290f2392e479ce740495a129019c488d8`.
- Before installing exact ship knobs, the launcher clears all ambient
  `LLAMA_ARG_*` and `GGML_SYCL_*` variables plus `LLAMA_API_KEY`,
  `LD_PRELOAD`, and `LD_AUDIT`. It also clears lock bypass variables and
  inherited lock functions, restores the canonical fleet lock paths, holds
  that lock for the complete server lifetime, and cleans flock/metadata on
  normal exit or post-lock startup failure.
- Production binds localhost and should be exposed only through a TLS reverse
  proxy or SSH tunnel. Direct LAN plain HTTP is explicit break glass: it
  requires `SERVE_ALLOW_INSECURE_NETWORK=1` plus an absolute, private,
  owner-controlled `SERVE_API_KEY_FILE`; inline keys cannot authorize a
  non-loopback bind. Wildcard CORS is rejected. Default logging is journald.
  `muse-b70.service` keeps that localhost/journal profile and caps restarts at
  three per five minutes (`RestartPreventExitStatus=2 3 75`). The template is
  not installed. Hydra surface `turbo:8095` is registered (candidate).
- `results/serve-release-history-quality-20260812T1752Z.md` is the canonical
  live release smoke. Its log identifies manifest `53e20fe1`, server
  `e5bfa43f`, SYCL `aa90882d`, model `7e9b74b7`, and the exact fixed
  configuration. It proves HTTP 401 without auth, untrusted-origin rejection,
  localhost-origin acceptance, authenticated root HTTP 404 (UI off), exact
  `MUSE-OK` / `SECOND-OK`, and turn-two cache reuse (`prompt_n=15`). Its
  authenticated `/apply-template` gate proves old and recent assistant
  `reasoning_content` markers render exactly once each, in order. It then
  echoes the actual turn-one `reasoning_content` **and** `content` into turn
  two and passes exactly; both requests use `max_tokens=128`. Client contract:
  preserve both assistant fields in history and budget enough output tokens
  for reasoning. `--reasoning-preserve` remains disabled/no-op for Muse.
  The older 17:16 `serve-final-quality` smoke ran mutable `build/bin` and did
  **not** record or validate the llama-server SHA; do not cite it as release
  integrity.
- Both no-GPU suites pass. `scripts/test-serve-preflight.sh` covers the fixed
  profile, positional/environment bypasses, manifested-DSO drift, keys,
  network/port/lock collision, and post-lock cleanup.
  `scripts/test-fullctx-server-compare.py` covers receipt provenance,
  control noise, speed, greedy/probability gates, and receipt immutability.
- Build gotcha: configure needs
  `-DLLAMA_BUILD_MTMD=OFF -DLLAMA_BUILD_MTMD_TOOL=ON` or llama-server fails
  to link (-lmtmd).
- Canonical measurements:
  short-prompt prefill **1277 ± 1.7 t/s** steady-state (req1 ~1026,
  one-time warmup), TTFT ≈ 380 ms fixed + ~1350 t/s marginal, and decode
  28.6 t/s at d0. At 131K, compact/ring SWA is a ~19.7 t/s warmed
  llama-bench estimate. Full-SWA is a fresh-server measurement and a
  different path: **10.592 t/s OFF → 19.032 ON (+79.681%)** on a frozen
  129024-token server fixture. Its full fixture prime is 503.19 t/s. Do not
  compare or report ring and full-SWA as the same mode.

## Kernel campaign result: reordered Q4_K small-N hoist shipped

`ggml/src/ggml-sycl/mmvq.cpp` now has an opt-in N=2..8 Q4_K reordered
specialization. It loads/unpacks each weight block and its scale/min metadata
once, then reuses them across the unrolled Q8_1 activation columns. Muse sets
`GGML_SYCL_LX_Q4_K_MMVQ_NCOLS_HOIST=1` in `serve-muse.sh`.

Warmed aggregate TG improved from 34.40/38.90/41.89 to
37.08/42.22/45.32 t/s at B=2/4/8 (about +8%); B=1 and prefill stayed flat.
Gates: exact Muse-shape ops 21/21, KLD 0.000499 / 99.191%, stable concurrent
server greedy byte- and token-identical in 8/8 slots. Receipts and the natural
text scheduler-nondeterminism analysis are in `results/kernel-20260812/` and
`RESULTS.md`. A 4/16/32-subgroup packing sweep was flat-to-worse and removed.

## Kernel campaign result: full-SWA masked-prefix decode skip shipped

`GGML_SYCL_LX_FATTN_DECODE_MASK_BOUNDS=1` makes VEC decode derive the live
KV tile bounds from the attention mask and skip tiles that are entirely
`-inf`. It retains the full physical SWA KV/history, so edited turns still
reuse cache to their divergence point; it changes traversal, not model
semantics. On B70, a 1024-thread cooperative scanner reduces mask discovery
work further, with the original 128-thread path retained as the device
fallback. `GGML_SYCL_LX_FATTN_PARALLEL_BLOCKS=24` is the best one-seat
full-depth split width.

The first fresh-server A/B/A is retained as intermediate validation of the
earlier 128-thread scanner. All
arms used `--swa-full -c 131072 -np 1` and a frozen 129024-token fixture
(token SHA-256 `893c12231ab1360f5f1d6ff0b78ab7568142e856b5526c27fd323a32dd35872e`):
OFF-A 10.5875, ON 17.2918, OFF-B 10.5966 t/s. The OFF mean is 10.5921,
earlier-scanner gain +63.252%, and control spread 0.086%. Receipt:
`results/fullctx-server-final-20260812T1605Z/`.

The final clean wide-scanner server receipt is **19.0319 t/s** sampled and
**19.3986 t/s** greedy, or **+79.6806%** over the same OFF mean; its 129024
token prime took 256.46 s / 503.19 t/s. Greedy tokens and content match both
OFF controls exactly. The noise-calibrated probability gate passes, and the
broad compile-regression KLD gate passes at 0.000494 mean / 99.240% same-top.
Receipt:
`results/fullctx-server-wide-final-20260812T163904Z/`.

Officially sampled tails are not bit-identical, but neither are OFF-A and
OFF-B; sampled exactness remains diagnostic rather than a fake pass/fail
gate. The TILE prefill bounds experiment passed its targeted op test but
improved only ~0.7% through 40K of a real prompt; it was aborted, reverted,
and is not shipped.

## The three traps that will bite you if you skip this section

1. **llama-bench lies about prefill on reorder builds.** Weights reorder for
   MMVQ on FIRST DECODE; llama-bench pp tests run before any decode → dense
   GEMM → fast (930). A real server decodes at warmup → every wide prefill
   crawls through 8-col reorder-MMVQ (61 t/s). Fix (shipped):
   `GGML_SYCL_LX_REORDER_MULTICOL_MKL=1`. Corollary: llama-perplexity never
   decodes either, so KLD is equally blind to this path — gate it with
   server-level greedy A/B. Measure serving-path prefill changes at the
   server or with llama-batched-bench warmed rows (`-npl 1,1,...`, skip row 1).
2. **One Level-Zero client at a time** or the xe driver wedges
   (lx/env.sh). GPU lock: `lx/scripts/with-gpu-lock` /
   `lib-gpu-lock.sh`; `scripts/bench-sweep.sh` here takes it. Never serve
   Muse alongside Laguna :8092 / treebeard :8093-4.
3. **The old 19.36 t/s full-depth receipt was not `--swa-full`.**
   `llama-bench` defaults to compact/ring SWA; its repeated full-SWA depth
   samples are also invalid after the first live sample because sequence
   save/restore discards masked history. Use the fresh-server full-context
   harness for full-SWA claims. Ring/default is the fastest linear-history
   mode; full-SWA buys branch/edit cache reuse.

## Remaining campaign: stronger batched small-N decode on SYCL

The single remaining lever, and it unlocks THREE things at once:
- `-np` scaling: the hoist raises decode aggregate scaling from 1.47× to
  about 1.59× at 8 slots (28.4 → 45.2 t/s), but it still fails to amortize
  most of the weight traffic across sequences.
- Speculation: ngram self-spec is a net LOSS (18.3 vs 28.2 t/s at 16%
  accept) because verifying k draft tokens costs ~k single-token passes —
  same root cause. Fix the kernel, re-run `results/spec/` A/B and it
  plausibly flips to a win (decode is bandwidth-bound at 446 GB/s; batch
  verify should be nearly free).
- Single-stream decode stays walled at 28.6 (15.6 GB × 28.6 ≈ full
  bandwidth) — batching is the only way more tokens/s comes out of this
  card without a smaller quant.

Next credible design (all on this box):
- The old np12 wide-MMVQ, MMID-batch, K-split, column-multipass, and subgroup
  packing ideas were audited or measured and do not solve Muse N=2/4/8.
- If another round is justified, extract the fused Q4_K dequant/XMX tile from
  historical commit `05755f5f30` into an ordinary dense `MUL_MAT` kernel and
  design a small-N tile that shares dequantized weights without padding all
  batches to a wasteful 16 columns.
- Instrument: `llama-batched-bench -npl 1,1,2,4,8` warmed rows for the knee;
  `b70-kernel-trace --mode ggml` on an ntg-heavy batched run to see which
  mul_mat variant handles batch 2-8 decode (expect the MMVQ family, not
  dense GEMM); muse Q4_K weight shapes are ne=[6656,4096/256/19968].
- Gates: rig rules in `lx/AGENTS.md` (lock, golden, KLD or server-greedy
  for reorder-affected paths, never promote ungated).

## Campaign history

`RESULTS.md` is the full story (sweeps → union build → KLD → chat parser →
the 21× prefill finding → --swa-full → MMQ/DMMV/speculation negatives →
TTFT curve). `README.md` is the operator view. Bench receipts in
`results/`, sweep tool `scripts/bench-sweep.sh` (takes the lock, loader
assert, JSON out), smoke `scripts/smoke-test.sh`. Session memory is in
`~/.claude/projects/-home-frosty40-turbo-muse/memory/`.

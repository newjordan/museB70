# Muse Glimmer 30B on B70 — serving-config campaign receipts

Campaign 2026-08-11/12. All benches under the lx GPU lock via
`scripts/bench-sweep.sh` (ab-crossmodel.sh pattern: loader assert, JSON
receipts in `results/sweep-20260811/`). Model:
`/mnt/data2tb/benchmodels/muse-glimmer-30B-kquant-17gb.gguf` (Q4_K_M, 15.6 GB).

## Binaries

| name | tree | so_sha | note |
|---|---|---|---|
| champ | worktrees/champ-muse-probe (c7d3bfe6d + muse patch) | 3091e228 | crossmodel probe |
| base | worktrees/base-muse-probe (97cb79588 + muse patch) | e0cd4d59 | control |
| muse-serve (pre-bounds) | worktrees/muse-serve (Muse + Q4_K small-N patches) | ef18f65a | prior promoted build |
| earlier bounds scanner | build-smalln at c31785423 + mask-bounds patch | c9720e6c | 17.292 t/s intermediate A/B/A |
| final gated build | build-smalln, code committed as 3ce44d373 | d1eac747 | 19.032 t/s server + KLD gates |
| final promoted build | build/bin at 3ce44d373 | aa90882d | promotion source; mutable, not served directly |
| **production release** | releases/muse-serve-3ce44d373 | aa90882d | **ship**; fixed manifest `53e20fe1`, server `e5bfa43f`, all local DSOs covered |

## Stage 1+2 — knob sweep (champ binary, d0)

| leg | pp512 | pp4096 | tg128 |
|---|---|---|---|
| control (DNN off, graph off, ub2048/b2048, f16 KV) | 933.1 ±2.9 | 1322.3 ±1.1 | 28.62 ±0.03 |
| oneDNN on | 927.7 | 1315.7 | 28.66 |
| ub4096/b4096 | 930.1 | **1336.1** | 28.64 |
| ub1024 | 925.8 | 1182.3 | 28.56 |
| SYCL graph on | 927.5 | — | 28.63 |

Crossmodel parity: control reproduces the 2026-08-11 crossmodel receipts
(925.6/28.6). **Decode = 15.6 GB × 28.6 t/s ≈ 446 GB/s: the memory-bandwidth
wall.** No kernel knob moves tg at d0. oneDNN and graph: keep OFF.

## Depth (compact/ring SWA; historical llama-bench receipt)

| leg | binary + env | pp2048@d32768 | tg64@d32768 |
|---|---|---|---|
| scout-champ | champ, knobs off | 1045.9 ±1.2 | 25.37 ±0.03 |
| scout-base | base, stock | 1044.8 ±1.1 | 25.14 ±0.02 |
| u-fattn8 | muse-serve, split-K 8 | 1044.1 | 25.15 |
| u-fattn16 | muse-serve, split-K 16 | 1043.7 | 25.49 |
| **u-fattn32** | muse-serve, split-K 32 | 1044.5 | **25.52 ±0.01** |
| u-mc | muse-serve, multicol-MKL | 1045.2 | 25.28 |

tg128 depth curve (champ, r3): 28.61 @d0 → 27.04 @d8192 → 25.17 @d32768
(−12% at 32K — mild, thanks to SWA + 2 KV heads).

These rows use llama-bench's default compact/ring SWA cache. They do not
measure `--swa-full`; that distinction matters sharply at 129K.

Union parity (knobs off): pp512 931.8 / tg128 28.68 ≡ champ. ✔

## Reading

1. **The champion/lx stack is dormant on dense Muse** — no Laguna-class wins
   exist here. Base ≈ champ ≈ union everywhere.
2. At this sweep stage, the only live lx knob was
   `GGML_SYCL_LX_FATTN_PARALLEL_BLOCKS=32`:
   +0.6% decode at 32K depth (25.52 vs 25.37), monotone in width
   (8 < 16 < 32), sd 0.01. It was selected at this stage and gated by the
   KLD check below; the later full-SWA d131K sweep selected width 24.
3. Ship flags: `-ub 4096 -b 4096` (+1% pp4096, flat elsewhere; VRAM headroom
   is ample), f16 KV at full 131072 ctx (compact/ring SWA makes it ~2 GiB),
   `-fa on`, DNN off, graph off.

## Quality gate

kld-crossmodel.sh instrument (16×512 wikitext-2, bars: mean KLD ≤ 0.010,
same-top ≥ 99.0%): base-muse-probe capture → muse-serve compute with the full
ship env (incl. fattn split-K 32).

RESULT (2026-08-12 04:14Z): **PASS** — mean KLD **0.000487 ± 0.000047**
(bar ≤ 0.010, 20× margin), same-top **99.289 ± 0.132 %** (bar ≥ 99.0%).
Logs: lx/results/crossmodel-20260811/kld-muse-serve-ship-{capture,compute}.log
(capture .kld deleted after compute; 1.6 GB). The split-K 32 divergence is
reduction-order-only and well inside campaign bars → shipped ON.

## Initial ship config (before the full-SWA decode campaign)

`serve-muse.sh`: muse-serve binary, port 8095, `-c 131072 -np 1 -fa on
-ctk f16 -ctv f16 -ub 4096 -b 4096 -t 16 --jinja` (embedded template),
temp 1.0 / top-p 0.95 / top-k 64 (official card), env block per RESULTS
above.

Historical full-depth compact/ring validation (ship env + ub4096, r2):
**pp2048@d131072 676.6 ±8.1 t/s, tg64@d131072 19.36 ±0.00 t/s**. The
19.36 result was never a `--swa-full` measurement; current warmed
llama-bench ring runs are ~19.7 t/s. See the fresh-server full-SWA A/B/A
below.

## Chat parser (2026-08-12)

The src/-only Muse pick left out upstream 62bf73d25's `common/chat.cpp`
specialized parser ("Muse Glimmer format: ' to=<recipient>' recipients,
<|eom|>/<|eot|> terminators"). Without it the channel headers leak into
`content` ('to=user<|message|>SECOND-OK'). Applied as
`patches/muse-chat-parser-common.patch` (+ common/speculative.cpp tweak),
llama-server rebuilt (so_sha unchanged f8da4229 — kernels untouched).

Smoke (results/smoke-20260811.md): exact-content pass, reasoning separated
into `reasoning_content` (channel-style thinking model), multi-turn cache
reuse confirmed (turn-2 prompt_n=27, no re-prefill), n_ctx_slot=131072,
serving decode 28.8-33.1 t/s.

## FINDING 2026-08-12 — llama-bench masks the reorder-MMVQ prefill collapse; serving needs REORDER_MULTICOL_MKL=1

**Real server prefill was 61 t/s, not the 930 t/s llama-bench reported — a
15× gap fixed by one env var.**

Chain of evidence (results/npl-knee*.txt, pp-server.log, ktrace-{fast,slow}/):
1. llama-batched-bench per-sequence prefill ≈ 61 t/s (B=1 first row 108, all
   warmed rows 61) while llama-bench pp512 = 930. FA on/off/auto ruled out
   (928/883/930). kv-unified, --swa-checkpoints 0, --swa-full: all still 61.
2. llama-server confirmed reality: 2664-token prompt → **61.1 t/s** (43.6 s!),
   cold and warm, `cache_prompt:false`.
3. GGML_SYCL_DEBUG dispatch diff: slow path routes matmuls through the
   generic `ggml_sycl_op_mul_mat` (reorder-MMVQ family); fast path uses
   `mul_mat_sycl` dense GEMM only.
4. Mechanism: the champion build **reorders weights for MMVQ on first
   decode**. llama-bench pp tests run before any decode → dense GEMM → fast.
   Any real server decodes immediately, weights reorder, and every later
   wide prefill batch crawls through 8-col reorder-MMVQ.
5. Fix = the lx stack's wide-batch fall-through:
   `GGML_SYCL_LX_REORDER_MULTICOL_MKL=1` → server prefill **1277 ± 1.7 t/s
   steady-state** (canonical receipt: 6 identical uncached 2664-tok requests,
   fresh server, results/canonical-pp.log — req1 1026 t/s carries ~500 ms
   one-time warmup, req2-6 = 1277). **61 → 1277 = 20.9×.** Same mechanism as
   Laguna's FINDING_20260811_reorder_multicol_c4 (307→1726 t/s), now shown
   to apply to dense foreign models too. (The 61 is NOT warmup: stable
   cold+warm across 4 server restarts; the server's load-time warmup decode
   triggers the reorder before request #1.)

**Instrument lesson (generalizes):** llama-bench structurally cannot see any
pathology triggered by decode-then-prefill ordering, and llama-perplexity
never decodes either — so both the speed receipts AND the KLD instrument are
blind to reordered-weight prefill paths. Serving-path knobs must be measured
at the server (or llama-batched-bench with a warmed first row).

Quality gate for the knob: server-level greedy A/B (2664-token wikitext
prompt, temp 0, 64 tokens), mc-off vs mc-on: **token-identical**
(results/greedy-mc-{off,on}.txt). fattn32 already KLD-gated above.

## Multi-slot knee (batched-bench, fix ON, npp512/ntg128)

| slots | S_PP agg | S_TG agg | TG per-stream |
|---|---|---|---|
| 1 | 889 | 28.4 | 28.4 |
| 2 | 1106 | 34.4 | 17.2 |
| 4 | 1276 | 38.9 | 9.7 |
| 8 | 1345 | 41.8 | 5.2 |

Prefill scales; decode barely does (1.47× at 8 slots — the 8-col MMVQ decode
path doesn't batch across sequences on this backend). **-np 1 stays the ship
default**; -np 2 is the only sane multi-user point (+21% aggregate, −39%
per stream). Kernel-level multi-slot decode batching is the standout future
opportunity.

## FINDING 2026-08-12 — Q4_K reordered small-N weight hoist adds 7-9% decode

The reordered Q4_K multi-column MMVQ was already one kernel for N=2..8,
but its generic column loop re-entered the Q4_K dot helper for every column.
The new opt-in specialization loads/unpacks each Q4_K weight block and its
scale/min metadata once, then reuses those values across the unrolled Q8_1
activation columns. Ship knob:
`GGML_SYCL_LX_Q4_K_MMVQ_NCOLS_HOIST=1`.

Fresh-process warmed knees (same `build-smalln` binary, feature OFF vs ON;
receipts under `results/kernel-20260812/`):

| slots | TG OFF | TG ON | change | PP effect |
|---|---:|---:|---:|---:|
| 1 | 28.34 | 28.44 | flat | flat |
| 2 | 34.40 | 37.08 | +7.8% | flat |
| 4 | 38.90 | 42.22 | +8.5% | flat |
| 8 | 41.89 | 45.32 | +8.2% | flat |

Gates:

- Exact SYCL-vs-CPU `MUL_MAT` cases for Muse `k=6656`, output rows
  256/4096/19968, and N=2..8: **21/21 PASS** without loosening tolerance.
- Broad 16-chunk Muse KLD: **0.000499 mean / 99.191% same-top — PASS**
  (bars 0.010 / 99.0%).
- Eight simultaneous greedy server requests on the previously stable
  repetitive workload: control A/A, rebuilt feature-OFF, and feature-ON were
  byte- and token-identical in **8/8 slots for 64 tokens**.
- Natural-text concurrent requests reproduced the documented base
  nondeterminism: six unique outputs in both arms and 6/8 candidate outputs
  exactly matched a control output in a different slot. It is retained as a
  diagnostic receipt, not misreported as a kernel failure or pass.

Packing 4/16/32 subgroups per workgroup was flat-to-worse; the existing 8
was retained and the experimental geometry knob removed. B=8 scaling rises
from 1.47× to about 1.59×, still not enough to change the `-np 1` default.

## --swa-full A/B (loop iteration 2, 2026-08-12)

Instrument: server-level branch-edit test — prompt X+Y+Z (2664 tok), resend
identical, then resend with Y edited early (divergence ~tok 2022).

| config | full prefill | identical resend | branch (edit mid-history) |
|---|---|---|---|
| default SWA cache | 2664 tok @ 988 t/s | prompt_n=1 | **prompt_n=2664** (2204 ms — full re-prefill) |
| --swa-full | 2664 tok @ 1029 t/s | prompt_n=1 | **prompt_n=642** (750 ms — reuse to divergence) |

Reading: with the default SWA cache, ANY edit/regeneration re-prefills the
whole conversation (at 131K depth that's ~2-3 min of wall clock at the
at-depth prefill rate). --swa-full keeps full KV for the 39 SWA layers
(+~5.1 GiB at 131072 ctx — fits fine at -np 1: ~25 of 30.3 GiB total) and
restores divergence-point reuse. This test established the cache behavior,
but its claim that decode speed was otherwise unchanged was wrong at maximum
depth: full-SWA scans masked historical KV unless the bounds optimization
below is enabled. The production launcher now fixes full-SWA; compact/ring
tests require a separate experimental/manual launcher.

## FINDING 2026-08-12 — skip the masked full-SWA prefix during VEC decode

With full-SWA, 39/52 Muse layers retain old SWA KV for branch/edit reuse, but
the VEC decode kernel still walked tiles that the attention mask immediately
turned into `-inf`. `GGML_SYCL_LX_FATTN_DECODE_MASK_BOUNDS=1` derives the
live tile-aligned KV range from that mask and fast-forwards each split over
fully masked prefix tiles. The physical cache and mask semantics are
unchanged. The final B70 path scans the mask cooperatively with a 1024-thread
workgroup; devices without that workgroup size retain the 128-thread
fallback. Split width 24 was best for one-seat full-depth service.

### Intermediate 128-thread scanner validation

The first server A/B/A used a fresh process per arm with `--swa-full`,
`-c 131072 -np 1 -b 4096 -ub 4096`, f16 KV, and the frozen 129024-token
fixture whose token SHA-256 is
`893c12231ab1360f5f1d6ff0b78ab7568142e856b5526c27fd323a32dd35872e`.
Each measured continuation reused 129023 cached tokens and evaluated one
prompt token. Receipt: `results/fullctx-server-final-20260812T1605Z/`.

| arm | bounds | sampled 256-token decode | greedy sidecar |
|---|---:|---:|---:|
| OFF-A | 0 | 10.5875 t/s | 10.8197 t/s |
| earlier scanner ON | 1 | **17.2918 t/s** | **17.5838 t/s** |
| OFF-B | 0 | 10.5966 t/s | 10.8215 t/s |

OFF mean 10.5921 t/s, OFF spread 0.086%, earlier-scanner gain **+63.252%**.
This receipt remains useful validation, but it is not the final ship
headline.

The intermediate quality accounting also exposed process-level sampled
nondeterminism:

- Greedy token sequences are identical across OFF-A, ON, and OFF-B.
- Officially sampled tails are not exact, but OFF-A and OFF-B already diverge
  at token 9. ON diverges from OFF-A at token 9 and OFF-B at token 23, so
  sampled exactness is not a stable process-level gate.
- Raw probability payloads are not bit-exact between OFF controls either.
  Mean JS divergence is 0.000394 for OFF-A/OFF-B versus 0.000441 and
  0.000507 for the two OFF/ON comparisons: the same small numerical class,
  not evidence of a semantic quality shortcut.

### Final clean wide-scanner promotion

The final build was run in a fresh server against the same frozen fixture and
OFF controls. Each continuation reused 129023 cached tokens and evaluated
one prompt token.

| arm | sampled 256-token decode | greedy/logprob sidecar |
|---|---:|---:|
| OFF mean | 10.5921 t/s | 10.8206 t/s |
| final wide scanner | **19.0319 t/s** | **19.3986 t/s** |

The sampled serving gain is **+79.6806%**. The final 129024-token prime took
256.46 s / 503.19 t/s. Receipt:
`results/fullctx-server-wide-final-20260812T163904Z/`. This is the canonical
full-SWA ship result. The historical 19.36 t/s number above is compact/ring
SWA and is not a valid full-SWA baseline.

Final quality gates:

- Greedy token IDs and content are exactly equal across final, OFF-A, and
  OFF-B.
- The probability gate passes against measured OFF/OFF arithmetic noise.
  Worst candidate mean top-N TV proxy is 0.0061813 versus control 0.0062329;
  max TV proxy is 0.018135 versus 0.016477; max selected-logprob delta is
  0.051246 versus control 0.030084 and remains below the 2x control allowance
  of 0.060169.
- Broad 16-chunk KLD passes at **0.000494 mean / 99.240% same-top** (bars
  0.010 / 99.0%). Logs:
  `lx/results/crossmodel-20260811/kld-muse-fullctx-bounds-final-20260812T163521Z-{capture,compute}.log`.
  KLD is the broad compile-regression gate; it does not activate the
  full-depth mask-bounds path, which is covered by the server greedy and
  probability gates above.

The final source is commit `3ce44d373`. The gated build-smalln library SHA is
`d1eac74749440cca57b7e43824d8c57845cc77a0e1da2930b8354f38ba6e5b2c`;
the promoted build/bin SHA is
`aa90882dc06d653c3f7a580ff6e1f34a5d84e02225eebcbf92cf243c0db69f06`.
They differ in build-directory metadata; their linked executable code, data,
and SYCL offload sections are identical.

A TILE prefill bounds specialization passed its targeted SYCL-vs-CPU op
case, but a real 129K prompt screen was only ~0.7% faster through 40K. The
run was stopped early, the TILE code was reverted, and no result from that
incomplete screen is promoted or claimed.

## Decode kernel knobs + power profile (loop iteration 3, 2026-08-12)

MMQ/DMMV on the ship env (tg128 r5): control **28.64**, `GGML_SYCL_FORCE_MMQ=1`
28.58, `GGML_SYCL_DMMV_X=64` 28.58 — both flat-to-hair-worse. Ship env
unchanged; decode stays at the bandwidth wall.

b70-profile of the ship config (results/profile-{decode,prefill}/):

| mode | avg W (of 230 cap) | t/s | tok/joule |
|---|---|---|---|
| decode (tg) | 162.6 W (71%) | 28.6 | 0.18 (5693 J/1k-tok) |
| prefill (pp) | 129.2 W (56%) | 1335 | 10.33 (97 J/1k-tok) |

Reading: decode draws 71% of cap while stuck at 28.6 t/s — EUs busy but
waiting on memory (bandwidth-bound confirmed; not power/thermal-capped, so
no headroom via clocks either). Prefill is ~59× more energy-efficient per
token. The only remaining decode levers on this hardware are fewer
bytes/token (smaller quant — none on box) or speculation (next backlog item).

## Ngram self-speculation A/B (loop iteration 4, 2026-08-12) — NEGATIVE, ship without

Server-level, greedy, 256 tokens; workloads: natural wikitext ("nat") and a
highly repetitive markdown table ("rep"). Types tried: `ngram-mod` (defaults
never trigger — n_match=24; retried at n_match=8) and `ngram-simple`.
Receipts: results/spec/.

| leg | nat | rep | draft stats (rep) |
|---|---|---|---|
| baseline | 28.2 | 28.2 | — |
| ngram-mod defaults | 28.1 | 28.2 | never engaged |
| ngram-mod n_match=8 | 28.1 | **18.3** | 861 drafted / 137 accepted (16%) |
| ngram-simple | 28.1 | **19.3** | 852 / 74 (8.7%) |

Reading: speculation is a NET LOSS here even on its best-case workload, and
the mechanism is the same backend weakness the multi-slot knee exposed:
**small-batch decode barely scales** (batch-8 ≈ 1.47×), so verifying k draft
tokens costs nearly k single-token passes — the accept rate can't pay for
that. Ship config stays speculation-free. If the kernel-level batched-decode
gap ever closes (the standout lx follow-up), re-run this A/B — the same
property that fixes -np scaling makes speculation viable.

Side note: `--swa-full` (iteration 2) is also what makes server speculation
*possible* at all on this SWA model (spec requires seq-rm support; without
swa-full the server logs "speculative decoding not supported").

Also reproduced: base-vs-base greedy divergence on natural text after ~50
tokens (the documented master-series nondeterminism class,
FINDING_20260810_master_prefill_nondeterminism) — not knob-attributable;
rep workload was token-identical across all legs.

## Server-side TTFT curve + DNN/graph re-validation (loop iteration 5, 2026-08-12)

Because llama-bench is a masked instrument for prefill (see the FINDING
above), the oneDNN and SYCL-graph verdicts were re-taken at the server, plus
a TTFT curve (`/completion`, cache off, n_predict 16, fresh server per leg):

| leg | 16 tok | 64 | 256 | 1024 | 2664 | gen |
|---|---|---|---|---|---|---|
| ship | 783 ms* | 392 ms | 476 ms | 916 ms | 2132 ms (1250 t/s) | 28.8-30.5 |
| DNN on | 789 | 391 | 475 | 921 | 2136 | ≡ |
| graph on | 784 | 393 | 476 | 918 | 2136 | ≡ |

(*first request after model load carries ~400 ms extra one-time warmup.)

Fit: **~380 ms fixed cost per uncached request + ~1350 t/s marginal
prefill.** DNN and graph are confirmed no-ops on the real serving path —
ship env final. The 380 ms floor is per-request setup/launch-chain overhead
(kernel/upstream work, not reachable from runtime config).

## Production serving hardening closeout (2026-08-12)

The production path no longer executes mutable `build/bin`. It uses read-only
snapshot `releases/muse-serve-3ce44d373` (directories and executable/DSO
payloads mode 0555; release metadata mode 0444). The launcher's fixed expected
SHA256SUMS digest is
`53e20fe1250b7c2bcaf5dc4753b0ac80ed62469e58656758570436124dbd71e5`.
That manifest covers llama-server and every bundled local llama/ggml DSO;
startup verifies every entry and requires `ldd` resolution for all local DSOs
to remain inside the snapshot.

Validated identities:

| object | identity |
|---|---|
| source | commit `3ce44d373c3fe2b2f4c88f196f9d063d3eefb267` |
| release llama-server | SHA-256 `e5bfa43f97d5f4de00bc5ec51c73722a56e0c6e2ad544fd61b3f7c801201fa86` |
| release libggml-sycl | SHA-256 `aa90882dc06d653c3f7a580ff6e1f34a5d84e02225eebcbf92cf243c0db69f06` |
| model content | SHA-256 `7e9b74b7c8875e9e265695df9613bf6290f2392e479ce740495a129019c488d8` |
| model startup identity | `66308:45219851:16756681056:1786468158:1786468158` (`device:inode:size:mtime:ctime`) |

`serve-muse.sh --check` is the fast no-start/no-lock preflight: release and
DSO hashes, loader resolution, fixed config, network policy, and model stat
identity. `serve-muse.sh --verify-model` performs the same checks and adds the
full 15.6 GB model SHA-256. Positional llama-server passthrough is disabled.
The child cannot inherit hidden options: all ambient `LLAMA_ARG_*` and
`GGML_SYCL_*` variables, `LLAMA_API_KEY`, `LD_PRELOAD`, and `LD_AUDIT` are
cleared before the launcher installs only its exact knobs and validated auth.

The ship launcher fixes `-np 1 -c 131072 -ub 4096 -b 4096`, f16 KV,
full-SWA, and Web UI off. Deviations fail closed; benchmark/manual launchers
own experimental contexts, slot counts, cache modes, and UI. Production binds
localhost and is exposed remotely only through a TLS reverse proxy or SSH
tunnel. Direct LAN plain HTTP is break-glass only and requires both
`SERVE_ALLOW_INSECURE_NETWORK=1` and an absolute private
`SERVE_API_KEY_FILE`; a non-loopback inline key is rejected. Wildcard CORS is
also rejected.

The canonical fleet lock is non-bypassable from the caller: inherited lock
functions and bypass/path variables are cleared, canonical paths are restored,
and the lock is held for the server's entire lifetime. Traps terminate the
child and release flock/metadata; the regression suite also forces a
post-acquisition log-setup failure and proves cleanup. Default output is
journald. `muse-b70.service` preserves localhost/journald, restarts on runtime
failure after 10 seconds, prevents restart for validation/artifact/lock exits
2/3/75, and caps attempts at three in five minutes.

Canonical live receipt:
`results/serve-release-history-quality-20260812T1752Z.md` plus its `.log`.
The log records release `53e20fe1`, server `e5bfa43f`, SYCL `aa90882d`, model
`7e9b74b7`, `ctx=131072 np=1 b=4096 ub=4096 swa_full=1 pb=24
mask_bounds=1 host=127.0.0.1 auth=1 webui=0 cors=localhost`. Smoke gates are
exact: unauthenticated chat HTTP 401, untrusted origin rejected, localhost
origin allowed, authenticated root HTTP 404 (UI disabled), `MUSE-OK`,
and `SECOND-OK`. The authenticated
`/apply-template` gate supplies old `MUSE-OLD-REASONING-741` and recent
`MUSE-RECENT-REASONING-852` assistant `reasoning_content`; both markers occur
exactly once in the rendered prompt and in chronological order. This proves
template handling for old and recent reasoning. The end-to-end gate then
copies the actual turn-one response's `reasoning_content` and `content` into
the assistant history message for turn two; it returns exact `SECOND-OK` and
reuses cache with `prompt_n=15`. Both calls use `max_tokens=128` so the
reasoning model has adequate output budget.

Client contract: echo both assistant fields when extending history and budget
enough output tokens for reasoning. This is client-managed history;
`--reasoning-preserve` remains disabled/no-op for Muse and is not credited for
the result.

Correction: `results/serve-final-quality-20260812T171634Z.md` ran from mutable
`build/bin`. It logged the SYCL SHA/config but did not record or validate the
llama-server SHA, so it is a historical behavior smoke, not a release
integrity receipt.

Both no-GPU regression suites pass:

- `scripts/test-serve-preflight.sh`: invalid/fixed-profile settings,
  positional and ambient-env bypasses, representative manifested-DSO drift,
  key-file permissions/content/symlinks, network/CORS policy, port collision,
  lock bypass/collision, clean release, and post-lock failure cleanup.
- `scripts/test-fullctx-server-compare.py`: canonical receipt plus negative
  provenance, unstable-control, insufficient-speed, greedy-token, and
  probability-drift cases; source receipt hashes remain unchanged.

## Final ship config (campaign + loop closeout, 2026-08-12)

Everything below is server-validated with receipts in this file:

| axis | value | receipt |
|---|---|---|
| binary | read-only `releases/muse-serve-3ce44d373` from source `3ce44d373` | manifest `53e20fe1`; server `e5bfa43f`; SYCL `aa90882d`; every local llama/ggml DSO verified |
| context | fixed `-c 131072 -np 1`, f16 KV, --swa-full | branch A/B, VRAM ~25/30.3 GiB; launcher rejects deviations |
| batch | fixed `-ub 4096 -b 4096`, `-fa on` | sweeps + server curve; launcher rejects deviations |
| env | DNN off, graph off, FUSE_NORM_ROPE=1, kill-switches, REORDER_MULTICOL_MKL=1 (load-bearing), FATTN_PARALLEL_BLOCKS=24, FATTN_DECODE_MASK_BOUNDS=1, Q4_K_MMVQ_NCOLS_HOIST=1 | 16× prefill; small-N exact ops/KLD; bounds attention ops + full-context greedy/probability; broad compile-regression KLD; full-SWA decode +79.681% |
| sampling | temp 1.0 / top-p 0.95 / top-k 64 | official card |
| speculation | OFF | iteration-4 negative result |
| service safety | non-bypassable lifetime fleet lock + cleanup; localhost/API-only; sanitized llama/SYCL/loader env; no positional passthrough; model stat pin; journald | remote production via TLS proxy/SSH; direct LAN HTTP requires explicit break glass + private key file; systemd max 3 restarts/5 min |
| release smoke | `results/serve-release-history-quality-20260812T1752Z.md` | exact release/config IDs, auth/CORS/UI gates, old/recent template history once each, actual turn-one reasoning+content echoed, exact turn two, `prompt_n=15` |
| service regression | `scripts/test-serve-preflight.sh`; `scripts/test-fullctx-server-compare.py` | both no-GPU suites PASS: fixed config, full-manifest drift, env/lock/network bypasses and cleanup; provenance/control-noise/quality/speed negative cases |
| serving numbers | short prefill ~1250 t/s / TTFT 0.4-0.8 s; 129024-token full-SWA prime 503.19 t/s; decode 28-33 t/s short, 19.032 full-SWA @129K, ~19.7 compact/ring @131K | curves, final full-context server receipt, b70-profile |

Open kernel-level follow-up: the Q4_K hoist recovers about 8%, but batched
small-N decode is still only 1.59× at eight slots. A denser XMX/shared-tile
design would be needed to unlock strong `-np` scaling and speculation.

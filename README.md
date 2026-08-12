<p align="center">
  <img src="assets/brand/muse-b70-hero.png" alt="Muse B70" width="100%">
</p>

# muse

Muse Glimmer 30B (Q4_K_M, 15.6 GB) on Intel Arc Pro B70. One seat, full 131k context.

| | |
|---|---|
| decode @ 129k cached | **19.0 t/s** |
| full-ctx prime | **503 t/s** |
| short decode / prefill | 28.6 / ~1277 t/s |
| quality | exact greedy vs controls · KLD 0.000494 |

![full-context server bench](assets/bench/fullctx-server.svg)

## Install

Needs an Arc B70, Intel oneAPI, and the GGUF (not in git):

`/mnt/data2tb/benchmodels/muse-glimmer-30B-kquant-17gb.gguf`

```bash
git clone https://github.com/newjordan/museB70.git
cd museB70

gh release download v2026.08.12-b70 --repo newjordan/museB70 \
  --pattern 'muse-serve-3ce44d373-linux-b70.tar.zst'
sha256sum -c releases/ASSET_SHA256SUMS
tar --zstd -C releases -xf muse-serve-3ce44d373-linux-b70.tar.zst
rm -f muse-serve-3ce44d373-linux-b70.tar.zst
```

Skip the download if `releases/muse-serve-3ce44d373/bin/llama-server` is already there.

## Serve

```bash
./serve-muse.sh --check    # hashes + config, no GPU
./serve-muse.sh            # http://127.0.0.1:8095/v1
```

```bash
curl -s localhost:8095/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"muse-glimmer-30b-q4","messages":[{"role":"user","content":"hi"}],"max_tokens":128}'
```

- One GPU client at a time. The xe driver wedges if you stack this with Laguna, treebeard, or a bench.
- Localhost only. Use SSH or a TLS proxy for remote. Direct LAN HTTP is break-glass (`SERVE_ALLOW_INSECURE_NETWORK=1` + a mode-600 `SERVE_API_KEY_FILE`).
- Muse thinks in `reasoning_content`. Echo **both** that and `content` on later turns, and give it enough `max_tokens` (128 is the smoke default).
- Optional systemd template: `muse-b70.service` (not installed).

Fixed profile: `-np 1 -c 131072 -ub 4096 -b 4096 --swa-full`, f16 KV, API only. The launcher will not take other knobs.

## Bench

No GPU:

```bash
bash scripts/test-serve-preflight.sh
python3 scripts/test-fullctx-server-compare.py
```

Live smoke (takes the GPU lock):

```bash
bash scripts/smoke-test.sh
```

Re-run the published numbers (GPU lock, slower):

```bash
bash scripts/bench-sweep.sh ship -- -p 512 -n 128 -r 5
bash scripts/bench-fullctx-server.sh
```

Headline receipts: `RESULTS.md`. Raw files: `results/`. Kernel patches: `KERNEL_PROVENANCE.md`.

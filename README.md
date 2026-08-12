<p align="center">
  <img src="assets/brand/muse-b70-hero.png" alt="Muse B70" width="100%">
</p>

# muse

Muse Glimmer 30B (Q4_K_M, 15.6 GB) on Intel Arc Pro B70. One seat, full 131k context.
**Dense 28B**

Serving package: [Frosty40/Muse-Glimmer-30B-ArcB70-GGUF](https://huggingface.co/Frosty40/Muse-Glimmer-30B-ArcB70-GGUF)

| | |
|---|---|
| decode @ 129k cached | **19.0 t/s** |
| full-ctx prime | **503 t/s** |
| short decode / prefill | 28.6 / ~1277 t/s |

![full-context server bench](assets/bench/fullctx-server.svg)

## Install

```bash
git clone https://github.com/newjordan/museB70.git
cd museB70

hf download Frosty40/Muse-Glimmer-30B-ArcB70-GGUF muse-glimmer-30B-kquant-17gb.gguf

gh release download v2026.08.12-b70 --repo newjordan/museB70 \
  --pattern 'muse-serve-3ce44d373-linux-b70.tar.zst'
sha256sum -c releases/ASSET_SHA256SUMS
tar --zstd -C releases -xf muse-serve-3ce44d373-linux-b70.tar.zst
rm -f muse-serve-3ce44d373-linux-b70.tar.zst
```

## Serve

```bash
MODEL=muse-glimmer-30B-kquant-17gb.gguf \
LLAMA_BIN=./releases/muse-serve-3ce44d373/bin/llama-server \
  ./serve-muse-arc.sh
```

```bash
curl -s localhost:8095/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"muse-glimmer-30b-q4","messages":[{"role":"user","content":"hi"}],"max_tokens":128}'
```

Muse thinks in `reasoning_content`. Echo both that and `content` on later turns, and give it enough `max_tokens`.

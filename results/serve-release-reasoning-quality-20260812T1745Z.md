# Muse serve smoke — 2026-08-12T17:45:01Z

## browser origin policy
untrusted origin rejected; localhost origin allowed
Web UI disabled: authenticated root returned HTTP 404
old and recent assistant reasoning preserved once in rendered history

## authentication
unauthenticated chat HTTP 401; authenticated chat accepted

## turn 1 (short)
content: 'MUSE-OK'
reasoning (first 120ch): 'Reply with exactly: MUSE-OK\n\nWe need reply with exactly: MUSE-OK\n\nProbably just output MUSE-OK. Exactly. No extra whites'
prompt_n=64 prompt_ms=789 predicted_per_sec=28.6

## turn 2 (same conversation extended — cache reuse check: prompt_n should be small)
content: 'SECOND-OK'
prompt_n=21 prompt_ms=449 predicted_per_sec=33.1

## template sanity (server log lines)
0.12.653.757 I srv          init: chat template supports preserving reasoning, consider enabling it via --reasoning-preserve

## slots / ctx at load
0.12.650.370 I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 131072, kv_unified = 'false'

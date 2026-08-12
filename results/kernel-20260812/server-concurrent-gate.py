#!/usr/bin/env python3

import argparse
import json
import pathlib
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def wait_ready(base_url: str, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(base_url + "/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (OSError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"server did not become ready: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, required=True)
    parser.add_argument("--prompt-file", type=pathlib.Path, required=True)
    parser.add_argument("--prompt-chars", type=int, default=6000)
    parser.add_argument("--concurrency", type=int, required=True)
    parser.add_argument("--n-predict", type=int, default=64)
    parser.add_argument("--wait-ready", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.wait_ready:
        wait_ready(args.base_url, 180.0)

    prompt = args.prompt_file.read_text(encoding="utf-8")[: args.prompt_chars]
    request_common = {
        "prompt": prompt,
        "n_predict": args.n_predict,
        "temperature": 0,
        "top_k": 1,
        "top_p": 1.0,
        "min_p": 0.0,
        "seed": 42,
        "cache_prompt": False,
        "ignore_eos": True,
        "return_tokens": True,
        "stream": False,
    }
    (args.out_dir / f"{args.arm}-request.json").write_text(
        json.dumps(request_common, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    barrier = threading.Barrier(args.concurrency + 1)

    def request_slot(slot: int):
        payload = dict(request_common)
        payload["id_slot"] = slot
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            args.base_url + "/completion",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        barrier.wait()
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=600) as response:
            raw = response.read()
        result = json.loads(raw)
        return slot, time.monotonic() - started, raw, result

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(request_slot, slot) for slot in range(args.concurrency)]
        barrier.wait()
        results = [future.result() for future in futures]

    summary = []
    for slot, elapsed, raw, result in sorted(results):
        if result.get("id_slot") != slot:
            raise RuntimeError(f"slot mismatch: requested {slot}, got {result.get('id_slot')}")
        tokens = result.get("tokens")
        if not isinstance(tokens, list) or len(tokens) != args.n_predict:
            raise RuntimeError(f"slot {slot}: expected {args.n_predict} tokens, got {tokens!r}")
        content = result.get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"slot {slot}: response has no string content")

        stem = args.out_dir / f"{args.arm}-slot{slot}"
        stem.with_suffix(".raw.json").write_bytes(raw)
        stem.with_suffix(".json").write_text(
            json.dumps(result, sort_keys=True, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        stem.with_suffix(".tokens.json").write_text(
            json.dumps(tokens, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        stem.with_suffix(".content.txt").write_bytes(content.encode("utf-8"))
        summary.append({
            "slot": slot,
            "elapsed_s": elapsed,
            "n_tokens": len(tokens),
            "content_bytes": len(content.encode("utf-8")),
        })

    (args.out_dir / f"{args.arm}-summary.json").write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"arm": args.arm, "concurrency": args.concurrency, "slots": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
